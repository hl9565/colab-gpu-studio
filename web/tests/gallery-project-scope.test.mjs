import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  entryPoints: [fileURLToPath(new URL('../src/lib/gallery-scope.ts', import.meta.url))],
  bundle: true,
  format: 'esm',
  write: false,
});
const { projectGallery, projectViewer, runProjectAssetAction, startProjectGalleryLoad } = await import(
  'data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64')
);

const assetA = { id: 'image-a', project_id: 'project-a', mime: 'image/png' };
const assetB = { id: 'image-b', project_id: 'project-b', mime: 'image/png' };

test('switching projects hides old gallery and viewer and blocks old actions while loading and after failure', async () => {
  const requests = new Map();
  const errors = [];
  const apiCalls = [];
  let gallery = [];
  const deps = {
    loadGallery: (projectId) => new Promise((resolve, reject) => requests.set(projectId, { resolve, reject })),
    setGallery: (next) => { gallery = next; },
    onError: (cause) => errors.push(cause),
  };

  const stopA = startProjectGalleryLoad('project-a', deps);
  requests.get('project-a').resolve([assetA, assetB]);
  await Promise.resolve();
  assert.deepEqual(gallery, [assetA]);
  const viewer = projectViewer(assetA, 'project-a');
  assert.equal(viewer, assetA);
  stopA();

  const projectId = 'project-b';
  const stopB = startProjectGalleryLoad(projectId, deps);
  const deleteAsset = async (id) => { apiCalls.push(['delete', id]); };
  const updateAsset = async (id) => { apiCalls.push(['favorite', id]); };
  async function assertOldProjectIsBlocked() {
    assert.deepEqual(projectGallery(gallery, projectId), []);
    assert.equal(projectViewer(viewer, projectId), null);
    await runProjectAssetAction(assetA, projectId, () => deleteAsset(assetA.id));
    await runProjectAssetAction(assetA, projectId, () => updateAsset(assetA.id));
    assert.deepEqual(apiCalls, []);
  }

  await assertOldProjectIsBlocked();
  assert.deepEqual(errors, []);
  const failure = new Error('project-b gallery failed');
  requests.get(projectId).reject(failure);
  await assertOldProjectIsBlocked();
  assert.deepEqual(errors, [failure]);
  stopB();

  await runProjectAssetAction(assetB, projectId, () => updateAsset(assetB.id));
  await runProjectAssetAction(assetB, projectId, () => deleteAsset(assetB.id));
  assert.deepEqual(apiCalls, [['favorite', assetB.id], ['delete', assetB.id]]);
});

test('an empty project hides all assets and denies mutations', async () => {
  const unscopedAsset = { ...assetA, project_id: '' };
  assert.deepEqual(projectGallery([assetA, unscopedAsset], ''), []);
  assert.equal(projectViewer(unscopedAsset, ''), null);
  await runProjectAssetAction(unscopedAsset, '', async () => assert.fail('unscoped mutation'));
});

test('App guards gallery reuse before lookup, after lookup, and when restoring the draft', () => {
  const app = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');
  assert.match(app, /async function reuseAsset\(asset: GalleryAsset\) \{\s*const sourceProjectId = currentProjectId\.current;\s*if \(!belongsToProject\(asset, sourceProjectId\)\) return;[^]*?const job = await getGpuJob\(sourceJobId\);\s*if \(currentProjectId\.current !== sourceProjectId \|\| job\.project_id !== sourceProjectId\) return;\s*setAgainJob\(job\);/);
  assert.match(app, /useEffect\(\(\) => \{\s*if \(againJob && againJob\.project_id !== projectId\) \{\s*setAgainJob\(null\);\s*return;\s*\}\s*if \(againJob[^]*?setParams\(restored\);[^]*?setTab\('创作'\);\s*\}\s*\}, \[againJob, projectId, workflows\]\);/);
});
