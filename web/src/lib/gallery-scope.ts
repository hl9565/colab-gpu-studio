import type { GalleryAsset } from './gpu-api';

export function belongsToProject(asset: GalleryAsset, projectId: string): boolean {
  return Boolean(projectId && asset.project_id === projectId);
}

export function projectGallery(gallery: GalleryAsset[], projectId: string): GalleryAsset[] {
  return projectId ? gallery.filter((asset) => belongsToProject(asset, projectId)) : [];
}

export function projectViewer(viewer: GalleryAsset | null, projectId: string): GalleryAsset | null {
  return viewer && belongsToProject(viewer, projectId) ? viewer : null;
}

export async function runProjectAssetAction(asset: GalleryAsset, projectId: string, action: () => Promise<void>) {
  if (belongsToProject(asset, projectId)) await action();
}

export function startProjectGalleryLoad(projectId: string, {
  loadGallery, setGallery, onError,
}: {
  loadGallery: (projectId: string) => Promise<GalleryAsset[]>;
  setGallery: (gallery: GalleryAsset[]) => void;
  onError: (cause: unknown) => void;
}) {
  let alive = true;
  loadGallery(projectId).then((next) => {
    if (alive) setGallery(projectGallery(next, projectId));
  }).catch((cause) => {
    if (alive) onError(cause);
  });
  return () => { alive = false; };
}
