import argparse
import hashlib
import json
from pathlib import Path

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    args = parser.parse_args()
    team = args.training_root.resolve()
    if team.name != 'aic3':
        raise ValueError('Expected an aic3 training owner directory')
    from scale_fusion.workspace_state import CONFIG_PATH
    recipe = json.loads(CONFIG_PATH.read_text())
    names = ['vehicle_alignment', 'pyramid_joint']
    phases = {phase['id']: phase for phase in recipe['stages']}
    parent = None
    for name, expected_steps, expected_lineage in zip(names, (400, 1400), (400, 1800)):
        record = json.loads((team/'runs'/name/'training.json').read_text())
        if not (record['owner_team'] == 'aic3' and record['status'] == 'trained'
                and record['optimizer_steps'] == expected_steps
                and record['lineage_steps'] == expected_lineage
                and record['phase'] == phases[name]):
            raise ValueError('The complete fixed training stage differs: '+name)
        model = (team/record['checkpoint']).resolve()
        if not model.is_relative_to(team) or digest(model) != record['checkpoint_sha256']:
            raise ValueError('The own model checkpoint differs: '+name)
        if parent is None:
            if record['initialization']['public_sha256'] != recipe['initialization_sha256']:
                raise ValueError('The public initialization differs')
        elif record['initialization']['own_parent_sha256'] != parent:
            raise ValueError('The own two-stage parent chain differs')
        parent = record['checkpoint_sha256']
    descriptor = dict(owner_team='aic3', model_file=str(model.relative_to(team)),
        checkpoint_sha256=parent, lineage_steps=1800, protocol=recipe['inference'],
        single_checkpoint=True, stages=names,
        public_initialization_sha256=recipe['initialization_sha256'])
    target = team/'runs/retrained_model_descriptor.json'
    temporary = target.with_name(target.name+'.writing')
    temporary.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(target)
    print('Trained model:', model)
    print('RGB model descriptor:', target)

if __name__ == '__main__':
    main()
