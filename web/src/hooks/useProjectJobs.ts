import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import { apiMessage, getGpuJobs, type GpuJob } from '../lib/gpu-api';

export function createProjectJobsController(projectId: string, {
  loadJobs, setJobs, setRefreshingJobs, setJobError,
}: {
  loadJobs: (projectId: string) => Promise<GpuJob[]>;
  setJobs: (jobs: GpuJob[]) => void;
  setRefreshingJobs: (refreshing: boolean) => void;
  setJobError: (error: string) => void;
}) {
  let alive = true;
  let sequence = 0;
  const isCurrent = (request: number) => alive && request === sequence;

  async function refreshJobs() {
    if (!alive) return;
    const request = ++sequence;
    setRefreshingJobs(true);
    try {
      const next = await loadJobs(projectId);
      if (!isCurrent(request)) return;
      setJobs(next.filter((job) => job.project_id === projectId));
      setJobError('');
    } catch (cause) {
      if (isCurrent(request)) setJobError(apiMessage(cause));
    } finally {
      if (isCurrent(request)) setRefreshingJobs(false);
    }
  }

  return { refreshJobs, dispose: () => { alive = false; } };
}

export function useProjectJobs(projectId: string, loggedIn: boolean) {
  const [jobs, setJobs] = useState<GpuJob[]>([]);
  const [refreshingJobs, setRefreshingJobs] = useState(false);
  const [jobError, setJobError] = useState('');
  const active = useRef<{ projectId: string; refreshJobs: () => Promise<void> } | null>(null);

  useLayoutEffect(() => {
    setJobs([]);
    setJobError('');
    if (!loggedIn || !projectId) {
      setRefreshingJobs(false);
      return;
    }

    const controller = createProjectJobsController(projectId, {
      loadJobs: getGpuJobs, setJobs, setRefreshingJobs, setJobError,
    });
    active.current = { projectId, refreshJobs: controller.refreshJobs };
    void controller.refreshJobs();
    const timer = window.setInterval(() => void controller.refreshJobs(), 4000);
    return () => {
      controller.dispose();
      active.current = null;
      window.clearInterval(timer);
    };
  }, [projectId, loggedIn]);

  const refreshJobs = useCallback(async () => {
    const current = active.current;
    if (loggedIn && current?.projectId === projectId) await current.refreshJobs();
  }, [projectId, loggedIn]);

  return { jobs, setJobs, refreshingJobs, jobError, setJobError, refreshJobs };
}
