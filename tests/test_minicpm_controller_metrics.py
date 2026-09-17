from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from postraining.controller_metrics import write_controller_metrics


def test_direct_controller_events_keep_actor_warmup_and_evaluation_steps(tmp_path):
    rows = [dict(phase='warmup', step=0, warmups=n, accuracy=n / 64, time=100 + n)
            for n in range(1, 11)]
    rows += [dict(phase='train', step=n, warmups=10, accuracy=n / 32, time=200 + n,
                  component_std=0.25, vector_sigma=0.5, sampled_gaussian_actions=3,
                  sampled_epsilon_second_moment=1.5625, sampled_epsilon_rms=1.25,
                  sampled_noise_component_rms=0.3125, sampled_noise_vector_rms=0.625,
                  max_gaussian_kl=12.5, max_joint_kl=11.0, mean_joint_kl=4.0)
             for n in range(1, 11)]
    rows += [dict(phase='evaluation', step=n, warmups=10, accuracy=.5, time=300 + n)
             for n in (0, 5, 10)]
    logdir = tmp_path / 'events'
    with SummaryWriter(str(logdir)) as writer:
        for row in rows:
            write_controller_metrics(writer, row)
        # Events are flushed and readable while the training writer is still open.
        events = EventAccumulator(str(logdir)).Reload()
        actor = events.Scalars('rollout_quality/accuracy')
        assert [(event.step, event.value) for event in actor] == [(n, n / 32) for n in range(1, 11)]
        assert [event.step for event in events.Scalars('rollout_quality_warmup/accuracy')] == list(range(1, 11))
        assert [event.step for event in events.Scalars('evaluation/accuracy')] == [0, 5, 10]
        expected_metrics = {
            'policy_noise/component_std': 0.25,
            'policy_noise/vector_sigma': 0.5,
            'policy_noise/sampled_gaussian_actions': 3,
            'policy_noise/sampled_epsilon_second_moment': 1.5625,
            'policy_noise/sampled_epsilon_rms': 1.25,
            'policy_noise/sampled_noise_component_rms': 0.3125,
            'policy_noise/sampled_noise_vector_rms': 0.625,
            'kl/controller_gaussian_max': 12.5,
            'kl/controller_joint_max': 11.0,
            'kl/controller_joint_mean': 4.0,
        }
        for tag, value in expected_metrics.items():
            assert [(event.step, event.value) for event in events.Scalars(tag)] == [
                (n, value) for n in range(1, 11)
            ]
