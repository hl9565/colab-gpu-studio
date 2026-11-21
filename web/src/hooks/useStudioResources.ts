import { useCallback, useEffect, useRef, useState } from 'react';
import type { GpuAccount, GpuInstance } from '../components/AccountDirectory';
import { apiFetch, apiMessage, listGpuInstances } from '../lib/gpu-api';

export function uniqueBy<T>(items: T[], keyOf: (item: T) => string): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = keyOf(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function readAccounts(result: unknown): GpuAccount[] {
  if (Array.isArray(result)) return result;
  const data = result && typeof result === 'object' && 'data' in result ? result.data : undefined;
  return Array.isArray(data) ? data : [];
}

export function createStudioResourcesController({
  loadAccounts, loadInstances, setAccounts, setInstances, setResourceError,
}: {
  loadAccounts: () => Promise<unknown>;
  loadInstances: () => Promise<GpuInstance[]>;
  setAccounts: (accounts: GpuAccount[]) => void;
  setInstances: (instances: GpuInstance[]) => void;
  setResourceError: (error: string) => void;
}) {
  let alive = true;
  let sequence = 0;
  let inFlight = 0;

  async function loadResource<T>(load: () => Promise<T>) {
    inFlight += 1;
    try {
      return await load();
    } finally {
      inFlight -= 1;
    }
  }

  async function refreshResources({ source = 'manual' }: { source?: 'manual' | 'poll' } = {}) {
    if (!alive || (source === 'poll' && inFlight > 0)) return;
    const request = ++sequence;
    try {
      const [accounts, instances] = await Promise.all([loadResource(loadAccounts), loadResource(loadInstances)]);
      if (!alive || request !== sequence) return;
      const nextAccounts = uniqueBy(readAccounts(accounts), (account) => account.account_id);
      const nextInstances = uniqueBy(instances, (instance) => instance.instance_id || instance.id || '');
      setAccounts(nextAccounts);
      setInstances(nextInstances);
      setResourceError('');
    } catch (cause) {
      if (alive && request === sequence) setResourceError(`账号/实例：${apiMessage(cause)}`);
    }
  }

  return { refreshResources, dispose: () => { alive = false; } };
}

export function useStudioResources(loggedIn: boolean) {
  const [accounts, setAccounts] = useState<GpuAccount[]>([]);
  const [instances, setInstances] = useState<GpuInstance[]>([]);
  const [resourceError, setResourceError] = useState('');
  const refresh = useRef<(() => Promise<void>) | null>(null);

  useEffect(() => {
    if (!loggedIn) return;
    const controller = createStudioResourcesController({
      loadAccounts: () => apiFetch<unknown>('/api/gpu/accounts'),
      loadInstances: () => listGpuInstances() as unknown as Promise<GpuInstance[]>,
      setAccounts, setInstances, setResourceError,
    });
    refresh.current = controller.refreshResources;
    void controller.refreshResources();
    const timer = window.setInterval(() => void controller.refreshResources({ source: 'poll' }), 5000);
    return () => {
      controller.dispose();
      refresh.current = null;
      window.clearInterval(timer);
    };
  }, [loggedIn]);

  const refreshResources = useCallback(() => refresh.current?.(), []);
  return { accounts, setAccounts, instances, setInstances, resourceError, refreshResources };
}
