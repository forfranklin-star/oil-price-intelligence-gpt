from pathlib import Path

def test_no_runtime_simulation_calls():
    root=Path(__file__).parents[1]
    for p in root.rglob('*.py'):
        if p.name.startswith('test_'): continue
        text=p.read_text(encoding='utf-8',errors='ignore')
        assert 'np.random' not in text
        assert 'random.random' not in text
        assert 'interpolate(' not in text
        assert '.fillna(' not in text
        assert '.ffill(' not in text
        assert '.bfill(' not in text

def test_report_path():
    assert (Path(__file__).parents[1]/'reports').exists()
