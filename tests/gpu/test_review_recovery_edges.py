import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.db import Store
from app.gpu.compute import MockColabProvider, MockGPUExecutor
from app.gpu.mock_sessions import SessionError
from app.gpu.repository import Repository
from app.gpu.scheduler import Scheduler
from test_pr4_signoff_regressions import seed_execution_binding


def test_prepared_but_not_started_execution_recovers_same_identity():
    repo=Repository(Store('sqlite://'))
    job=repo.create('jobs','u',{'state':'loading','provider':'mock'})
    provider=MockColabProvider()
    handle=provider.create({},str(uuid.uuid4()),job['id'],str(uuid.uuid4()))
    lease=repo.create('leases','u',{'state':'active','owner_job_id':job['id'],'handle':handle})
    prepared=repo.prepare_execution(job['id'],'u',lease['id'],1)
    prepared=seed_execution_binding(repo,prepared,lease)
    executor=MockGPUExecutor()
    scheduler=Scheduler(repo,executor_factory=lambda job:executor)
    result=scheduler._load(prepared)
    assert result['state']=='running'
    assert executor.started==[prepared['execution_id']]
    assert len(repo.list('executions','u'))==1


def test_same_owner_foreign_lease_is_not_resolved():
    repo=Repository(Store('sqlite://'))
    lease=repo.create('leases','u',{'state':'active','owner_job_id':'other-job'})
    job=repo.create('jobs','u',{'state':'running','lease_id':lease['id']})
    assert Scheduler(repo)._lease(job) is None


def test_usage_concurrent_insert_has_one_database_identity(tmp_path):
    repo=Repository(Store('sqlite:///'+str(tmp_path/'usage.db')))
    payload={'job_id':'j','execution_id':'e','units':1}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:repo.record_usage_once('u',payload),range(2)))
    assert results[0]['id']==results[1]['id']
    assert len(repo.list('usage','u'))==1


def test_old_mock_poll_is_rejected_after_next_execution_starts():
    executor=MockGPUExecutor()
    handle={'runtime_name':'mock-runtime-'+str(uuid.uuid4())}
    first={'owner_id':'u','execution_id':str(uuid.uuid4())}
    second={**first,'execution_id':str(uuid.uuid4())}
    executor.start(handle,first,first['execution_id'])
    assert executor.poll(handle,first)['state']=='completed'
    executor.start(handle,second,second['execution_id'])
    with pytest.raises(SessionError,match='stale'):
        executor.poll(handle,first)


def test_retry_clears_old_presentation_and_archives_it():
    repo=Repository(Store('sqlite://'))
    job=repo.create('jobs','u',{'state':'failed','provider':'mock','stage':'sampling',
        'worker_timings':{'sampling':9},'output_bytes':123,'started_at':'old'})
    result=repo.retry_attempt(job['id'],'u','retry')
    for key in ('stage','worker_timings','output_bytes','started_at'):
        assert key not in result
    assert result['attempt_history'][0]['stage']=='sampling'


def test_handoff_refuses_successor_with_existing_lease():
    repo=Repository(Store('sqlite://'))
    first=repo.create('jobs','u',{'state':'cleaning_up','compatibility_signature':'sig'})
    following=repo.create('jobs','u',{'state':'waiting_for_capacity','compatibility_signature':'sig'})
    lease=repo.create('leases','u',{'state':'active','owner_job_id':first['id'],'deadline_epoch':100})
    other=repo.create('leases','u',{'state':'active','owner_job_id':following['id'],'deadline_epoch':100})
    assert not repo.handoff_lease('u',first['id'],following['id'],lease['id'],1)
    assert repo.get('leases',other['id'],'u')['owner_job_id']==following['id']
    assert repo.get('leases',lease['id'],'u')['owner_job_id']==first['id']


def test_raw_outputs_with_foreign_execution_are_rejected():
    repo=Repository(Store('sqlite://'))
    job=repo.create('jobs','u',{'state':'running','provider':'mock','execution_id':'new'})
    repo.create('leases','u',{'state':'active','owner_job_id':job['id'],'handle':{}})
    class Executor:
        def poll(self,handle,job):
            return {'state':'completed','execution_id':'old','outputs':[{'content':b'fake','mime':'image/png'}]}
    result=Scheduler(repo,executor_factory=lambda job:Executor())._run(job)
    assert result['state']=='cleaning_up'
    assert result['error_code']=='gpu_execution_id_mismatch'
    assert not result.get('outputs')


def test_retire_absent_placeholder_blocks_cross_owner_and_started_lease():
    repo = Repository(Store('sqlite://'))
    inst_a = repo.create('instances', 'u1', {'state': 'creating', 'provider': 'mock', 'current_job_id': 'job-a', 'budget_reserved': 5.0})
    lease_a = repo.create('leases', 'u1', {
        'state': 'creating', 'provider': 'mock', 'owner_job_id': 'job-a',
        'instance_id': inst_a['id'], 'runtime_id': 'rt-a', 'create_attempts': 1,
    })
    job_b = repo.create('jobs', 'u2', {
        'state': 'allocating', 'provider': 'mock', 'instance_id': inst_a['id'], 'lease_id': lease_a['id'],
    })
    retired, _ = repo.retire_absent_placeholder_for_retry(
        job_b['id'], 'u2', expected_lease_id=lease_a['id'], expected_generation=0, total_attempts=0, now=100.0
    )
    assert retired is False
    assert (repo.get('jobs', job_b['id'], 'u2').get('alloc_generation') or 0) == 0
    assert repo.get('instances', inst_a['id'], 'u1')['state'] == 'creating'
    assert repo.get('leases', lease_a['id'], 'u1')['state'] == 'creating'

    # Even same owner with create_attempts > 0 (started lease) must not be retired as pure placeholder.
    job_a = repo.create('jobs', 'u1', {
        'state': 'allocating', 'provider': 'mock', 'instance_id': inst_a['id'], 'lease_id': lease_a['id'],
    }, record_id='job-a')
    retired2, _ = repo.retire_absent_placeholder_for_retry(
        job_a['id'], 'u1', expected_lease_id=lease_a['id'], expected_generation=0, total_attempts=0, now=101.0
    )
    assert retired2 is False
    assert repo.get('leases', lease_a['id'], 'u1')['state'] == 'creating'


def test_unknown_create_online_recovers_without_delete_or_recreate():
    from app.gpu.compute import MockColabProvider

    repo = Repository(Store('sqlite://'))
    req_id = str(uuid.uuid4())
    rt_id = f'rt-alive-{uuid.uuid4().hex[:6]}'
    job = repo.create('jobs', 'u1', {'state': 'allocating', 'provider': 'mock', 'project_id': 'p1', 'snapshot': {'manifest': {'mock': True}}})
    lease = repo.create('leases', 'u1', {
        'provider': 'mock', 'state': 'creating', 'owner_job_id': job['id'],
        'request_id': req_id, 'runtime_id': rt_id, 'create_attempts': 1,
        'operation_name': 'operations/op-alive', 'handle': None,
    })
    job = repo.update('jobs', job['id'], {'lease_id': lease['id']}, 'u1')

    class AliveProvider(MockColabProvider):
        def inspect(self, handle):
            self.inspect_calls += 1
            return {'ready': True, 'deleted': False, 'allocation_failed': False, 'handle': {'runtime_name': f'runtimes/{rt_id}'}}

    provider = AliveProvider()
    sched = Scheduler(repo, provider_factory=lambda j: provider)
    # First R1 probe hits unknown or immediate inspect: let's set lease unknown from prior crash
    repo.update('leases', lease['id'], {'state': 'unknown', 'from_active_missing_handle': False}, 'u1')
    out = sched._allocation(repo.get('jobs', job['id'], 'u1'))
    assert out['state'] == 'preparing'
    assert provider.inspect_calls == 1
    assert provider.create_calls == 0
    assert provider.delete_calls == 0
    saved_lease = repo.get('leases', lease['id'], 'u1')
    assert saved_lease['state'] == 'active'
    assert saved_lease['handle']['runtime_name'] == f'runtimes/{rt_id}'


def test_post_rpc_create_cancelled_marks_cleanup_pending_with_handle():
    from app.gpu.compute import MockColabProvider

    repo = Repository(Store('sqlite://'))
    job = repo.create('jobs', 'u1', {'state': 'allocating', 'provider': 'mock', 'project_id': 'p1', 'snapshot': {'manifest': {'mock': True}}})

    class CancelDuringCreateProvider(MockColabProvider):
        def create(self, spec, request_id, job_id, runtime_id):
            h = super().create(spec, request_id, job_id, runtime_id)
            repo.request_job_cancel('u1', job_id, 100.0)
            return h

    provider = CancelDuringCreateProvider()
    sched = Scheduler(repo, provider_factory=lambda j: provider)
    after = sched._allocation(repo.get('jobs', job['id'], 'u1'))
    assert after['state'] in {'cleaning_up', 'cancelled'}
    leases = repo.list('leases', 'u1')
    assert len(leases) == 1
    assert leases[0]['handle'] is not None
    if after['state'] == 'cleaning_up':
        assert leases[0]['state'] in {'cleanup_pending', 'deleting', 'deleted'}
        final = sched._cleanup(after)
        assert final['state'] == 'cancelled'
    assert repo.list('leases', 'u1')[0]['state'] == 'deleted'


def test_commit_running_rejects_unpaired_start_intent_and_wrong_lease():
    repo = Repository(Store('sqlite://'))
    job = repo.create('jobs', 'u1', {'state': 'loading', 'provider': 'mock', 'project_id': 'p1'})
    lease1 = repo.create('leases', 'u1', {'state': 'active', 'provider': 'mock', 'owner_job_id': job['id'], 'handle': {'runtime_name': 'r1'}})
    lease2 = repo.create('leases', 'u1', {'state': 'active', 'provider': 'mock', 'owner_job_id': job['id'], 'handle': {'runtime_name': 'r2'}})
    prepared = repo.prepare_execution(job['id'], 'u1', lease1['id'], now=100.0)
    prepared = seed_execution_binding(repo, prepared, lease1)
    exec_id = prepared['execution_id']

    # Wrong lease must be refused without mutating the valid binding
    status_wrong, _ = repo.commit_running(job['id'], 'u1', lease_id=lease2['id'], execution_id=exec_id, now=101.0)
    assert status_wrong == 'stale'
    assert repo.get('jobs', job['id'], 'u1') == prepared

    # Missing/unpaired start_intent on fresh prepared job must be refused and fenced cleanly
    job2 = repo.create('jobs', 'u1', {'state': 'loading', 'provider': 'mock', 'project_id': 'p1'})
    lease3 = repo.create('leases', 'u1', {'state': 'active', 'provider': 'mock', 'owner_job_id': job2['id'], 'handle': {'runtime_name': 'r3'}})
    prepared2 = repo.prepare_execution(job2['id'], 'u1', lease3['id'], now=102.0)
    prepared2 = seed_execution_binding(repo, prepared2, lease3)
    repo.update('jobs', job2['id'], {'start_intent': {'execution_id': 'other-exec', 'state': 'starting'}}, 'u1')
    status_unpaired, _ = repo.commit_running(job2['id'], 'u1', lease_id=lease3['id'], execution_id=prepared2['execution_id'], now=103.0)
    assert status_unpaired == 'fenced'
    assert repo.get('jobs', job2['id'], 'u1')['state'] == 'cleaning_up'
