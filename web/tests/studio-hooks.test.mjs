import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const bundled = await build({
  stdin: {
    contents: `
      export { createStudioResourcesController } from './src/hooks/useStudioResources.ts';
      export { createProjectJobsController } from './src/hooks/useProjectJobs.ts';
    `,
    resolveDir: fileURLToPath(new URL('..', import.meta.url)),
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  define: { 'import.meta.env.VITE_API_BASE_URL': '""' },
  write: false,
});
const { createStudioResourcesController, createProjectJobsController } = await import(
  'data:text/javascript;base64,' + Buffer.from(bundled.outputFiles[0].text).toString('base64')
);

function controlledRequests() {
  const requests = [];
  const load = (...args) => new Promise((resolve, reject) => requests.push({ args, resolve, reject }));
  return { requests, load };
}

function resourceHarness() {
  const state = { accounts: [], instances: [], resourceError: '' };
  const accounts = controlledRequests();
  const instances = controlledRequests();
  const controller = createStudioResourcesController({
    loadAccounts: accounts.load,
    loadInstances: instances.load,
    setAccounts: (value) => { state.accounts = value; },
    setInstances: (value) => { state.instances = value; },
    setResourceError: (value) => { state.resourceError = value; },
  });
  return { state, accounts: accounts.requests, instances: instances.requests, ...controller };
}

function jobsHarness() {
  const state = { jobs: [], jobError: '', refreshingJobs: false };
  const { requests, load } = controlledRequests();
  const forProject = (projectId) => createProjectJobsController(projectId, {
    loadJobs: load,
    setJobs: (value) => { state.jobs = value; },
    setJobError: (value) => { state.jobError = value; },
    setRefreshingJobs: (value) => { state.refreshingJobs = value; },
  });
  return { state, requests, forProject };
}

test('resources accept both account shapes and keep the newer deduplicated snapshot', async () => {
  const { state, accounts, instances, refreshResources } = resourceHarness();
  const poll = refreshResources({ source: 'poll' });
  const manual = refreshResources();
  assert.equal(accounts.length, 2);
  assert.equal(instances.length, 2);
  const account = { account_id: 'account-a', alias: 'current' };
  const instance = { instance_id: 'instance-a', id: 'alternate-id' };
  const legacyInstance = { id: 'instance-b' };
  accounts[1].resolve({ data: [account, { ...account, alias: 'duplicate' }] });
  instances[1].resolve([instance, { instance_id: 'instance-a', id: 'different-id' }, legacyInstance, { ...legacyInstance }]);
  await manual;
  const expected = { accounts: [account], instances: [instance, legacyInstance], resourceError: '' };
  assert.deepEqual(state, expected);

  accounts[0].resolve([{ account_id: 'stale' }]);
  instances[0].resolve([{ id: 'stale' }]);
  await poll;
  assert.deepEqual(state, expected);

  const direct = refreshResources();
  accounts[2].resolve([account, account]);
  instances[2].resolve([instance]);
  await direct;
  assert.deepEqual(state, { accounts: [account], instances: [instance], resourceError: '' });
});

test('resources skip polling ticks while a slow refresh is pending and resume after it publishes', async () => {
  const { state, accounts, instances, refreshResources } = resourceHarness();
  const slow = refreshResources();
  await refreshResources({ source: 'poll' });
  const account = { account_id: 'account-a' };
  const instance = { id: 'instance-a' };
  accounts[0].resolve([account]);
  await refreshResources({ source: 'poll' });
  assert.equal(accounts.length, 1);
  assert.equal(instances.length, 1);
  assert.deepEqual(state, { accounts: [], instances: [], resourceError: '' });

  instances[0].resolve([instance]);
  await slow;
  assert.deepEqual(state, { accounts: [account], instances: [instance], resourceError: '' });

  const poll = refreshResources({ source: 'poll' });
  accounts[1].resolve([]);
  instances[1].resolve([]);
  await poll;
  assert.deepEqual(state, { accounts: [], instances: [], resourceError: '' });
});

test('resources skip polling while a failed refresh still has an outstanding sibling request', async () => {
  const { state, accounts, instances, refreshResources } = resourceHarness();
  const failure = refreshResources();
  accounts[0].reject(new Error('账号读取失败'));
  await failure;
  assert.equal(state.resourceError, '账号/实例：账号读取失败');
  await refreshResources({ source: 'poll' });
  assert.equal(accounts.length, 1);
  assert.equal(instances.length, 1);

  instances[0].resolve([]);
  await Promise.resolve();
  const poll = refreshResources({ source: 'poll' });
  accounts[1].resolve([]);
  instances[1].resolve([]);
  await poll;
  assert.equal(state.resourceError, '');
});

test('resources only publish the newest error and clear it on success', async () => {
  const { state, accounts, instances, refreshResources } = resourceHarness();
  const staleSuccess = refreshResources();
  const failure = refreshResources();
  accounts[1].resolve([]);
  instances[1].reject(new Error('实例读取失败'));
  await failure;
  assert.equal(state.resourceError, '账号/实例：实例读取失败');

  accounts[0].resolve([{ account_id: 'stale' }]);
  instances[0].resolve([{ id: 'stale' }]);
  await staleSuccess;
  assert.deepEqual(state, { accounts: [], instances: [], resourceError: '账号/实例：实例读取失败' });

  const staleFailure = refreshResources();
  const recovery = refreshResources();
  accounts[3].resolve([{ account_id: 'recovered' }]);
  instances[3].resolve([{ id: 'recovered' }]);
  await recovery;
  accounts[2].reject(new Error('过期账号请求'));
  instances[2].resolve([]);
  await staleFailure;
  assert.deepEqual(state, {
    accounts: [{ account_id: 'recovered' }], instances: [{ id: 'recovered' }], resourceError: '',
  });
});

test('jobs polling and manual refresh share sequence protection for list, error, and loading', async () => {
  const { state, requests, forProject } = jobsHarness();
  const { refreshJobs } = forProject('project-a');
  const poll = refreshJobs({ source: 'poll' });
  const manual = refreshJobs();
  assert.equal(requests.length, 2);
  const currentJob = { id: 'manual', project_id: 'project-a' };
  requests[1].resolve([currentJob]);
  await manual;
  const current = { jobs: [currentJob], jobError: '', refreshingJobs: false };
  assert.deepEqual(state, current);

  requests[0].resolve([{ id: 'older-poll', project_id: 'project-a' }]);
  await poll;
  assert.deepEqual(state, current);

  const staleFailure = refreshJobs({ source: 'poll' });
  const pendingManual = refreshJobs();
  requests[2].reject(new Error('过期轮询'));
  await staleFailure;
  assert.deepEqual(state, { ...current, refreshingJobs: true });
  requests[3].resolve([currentJob]);
  await pendingManual;
  assert.deepEqual(state, current);
});

test('jobs skip polling ticks while a slow refresh is pending without invalidating it', async () => {
  const { state, requests, forProject } = jobsHarness();
  const { refreshJobs } = forProject('project-a');
  const slow = refreshJobs();
  await refreshJobs({ source: 'poll' });
  await refreshJobs({ source: 'poll' });
  assert.equal(requests.length, 1);
  assert.deepEqual(state, { jobs: [], jobError: '', refreshingJobs: true });

  const job = { id: 'slow', project_id: 'project-a' };
  requests[0].resolve([job]);
  await slow;
  assert.deepEqual(state, { jobs: [job], jobError: '', refreshingJobs: false });

  const poll = refreshJobs({ source: 'poll' });
  requests[1].resolve([]);
  await poll;
  assert.deepEqual(state, { jobs: [], jobError: '', refreshingJobs: false });
});

test('jobs preserve the newest failure until a later refresh succeeds', async () => {
  const { state, requests, forProject } = jobsHarness();
  const { refreshJobs } = forProject('project-a');
  const poll = refreshJobs({ source: 'poll' });
  const manual = refreshJobs();
  requests[1].reject(new Error('任务读取失败'));
  await manual;
  requests[0].resolve([{ id: 'stale', project_id: 'project-a' }]);
  await poll;
  assert.deepEqual(state, { jobs: [], jobError: '任务读取失败', refreshingJobs: false });

  const retry = refreshJobs();
  const recovered = { id: 'recovered', project_id: 'project-a' };
  requests[2].resolve([recovered]);
  await retry;
  assert.deepEqual(state, { jobs: [recovered], jobError: '', refreshingJobs: false });
});

test('switching project disposes old requests and filters the new project response', async () => {
  const { state, requests, forProject } = jobsHarness();
  const previous = forProject('project-a');
  const oldPoll = previous.refreshJobs({ source: 'poll' });
  const oldManual = previous.refreshJobs();
  previous.dispose();

  const current = forProject('project-b').refreshJobs();
  await previous.refreshJobs();
  assert.deepEqual(requests.map(({ args }) => args), [['project-a'], ['project-a'], ['project-b']]);
  requests[1].reject(new Error('旧项目失败'));
  await oldManual;
  assert.deepEqual(state, { jobs: [], jobError: '', refreshingJobs: true });

  const foreignJob = { id: 'old-project', project_id: 'project-a' };
  const currentJob = { id: 'new-project', project_id: 'project-b' };
  requests[2].resolve([foreignJob, currentJob]);
  await current;
  requests[0].resolve([foreignJob]);
  await oldPoll;
  assert.deepEqual(state, { jobs: [currentJob], jobError: '', refreshingJobs: false });
});
