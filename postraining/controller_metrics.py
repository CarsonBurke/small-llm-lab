"""Direct controller event logging using the shared posttraining dashboard tags."""
from postraining.train_minicpm_vapo import _organized_tensorboard_tag

CONTROLLER_TAGS = {
    'max_gaussian_kl': 'kl/controller_gaussian_max',
    'max_joint_kl': 'kl/controller_joint_max',
    'mean_joint_kl': 'kl/controller_joint_mean',
    'behavior_logprob_max_error': 'replay/controller_logprob_max_error',
    'accepted_step_fraction': 'optimization/controller_accepted_step_fraction',
    'backtracks': 'optimization/controller_backtracks',
    'controller_loss': 'optimization/controller_score_loss',
    'controller_parameter_delta_norm': 'optimization/controller_parameter_delta_norm',
    'cycle_seconds': 'optimization/cycle_seconds',
    'elapsed_training_seconds': 'optimization/session_training_seconds',
    'total_training_seconds': 'optimization/total_training_seconds',
    'component_std': 'policy_noise/component_std',
    'vector_sigma': 'policy_noise/vector_sigma',
    'sampled_gaussian_actions': 'policy_noise/sampled_gaussian_actions',
    'sampled_epsilon_second_moment': 'policy_noise/sampled_epsilon_second_moment',
    'sampled_epsilon_rms': 'policy_noise/sampled_epsilon_rms',
    'sampled_noise_component_rms': 'policy_noise/sampled_noise_component_rms',
    'sampled_noise_vector_rms': 'policy_noise/sampled_noise_vector_rms',
}


def write_controller_metrics(writer, row):
    phase = row['phase']
    if phase not in {'train', 'warmup', 'evaluation'}:
        return
    step = int(row['warmups'] if phase == 'warmup' else row['step'])
    for name, value in row.items():
        if not isinstance(value, (int, float)):
            continue
        if phase == 'evaluation':
            tag = {'accuracy': 'evaluation/accuracy', 'correct': 'evaluation/correct',
                   'trajectories': 'evaluation/trajectories',
                   'evaluation_seconds': 'evaluation/seconds'}.get(name)
        else:
            tag = (_organized_tensorboard_tag('rollout', name)
                   or _organized_tensorboard_tag('train', name)
                   or CONTROLLER_TAGS.get(name))
            if tag and phase == 'warmup':
                category, suffix = tag.split('/', 1)
                tag = f'{category}_warmup/{suffix}'
        if tag:
            writer.add_scalar(tag, float(value), step, walltime=float(row['time']))
    writer.flush()
