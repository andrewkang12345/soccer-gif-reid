"""Regressions for auxiliary training seeing held-out players in another match."""
import hashlib
import json
from experiments.soccernet_full_v2 import train_v2


def write_roster(root, game, people):
    folder = root/'metadata'/hashlib.sha256(game.encode()).hexdigest()[:12]
    folder.mkdir(parents=True)
    (folder/'Labels-caption.json').write_text(json.dumps({
        'lineup': {'home': {'players': people}, 'away': {'players': []}}}))


def test_other_match_with_same_player_is_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v2, 'DATA', tmp_path)
    rows = [{'pid': 'held', 'name': 'Held Player', 'split': 'test', 'game': 'test_match'}]
    write_roster(tmp_path, 'another_match', [{'hash': 'held', 'name': 'Held Player'}])
    people = [{'person': 'anonymous_1', 'match': 'another_match'},
              {'person': 'anonymous_2', 'match': 'another_match'}]
    assert train_v2.identity_disjoint_ssl(people, rows) == []


def test_normalized_name_alias_is_also_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v2, 'DATA', tmp_path)
    rows = [{'pid': 'one_hash', 'name': 'Jose Player', 'split': 'val'}]
    write_roster(tmp_path, 'another_match', [{'hash': 'different_hash', 'name': 'José Player'}])
    assert train_v2.identity_disjoint_ssl([{'person': 'a', 'match': 'another_match'}], rows) == []


def test_unknown_roster_is_excluded_and_safe_match_is_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v2, 'DATA', tmp_path)
    rows = [{'pid': 'held', 'name': 'Held Player', 'split': 'test'}]
    write_roster(tmp_path, 'safe_match', [{'hash': 'train', 'name': 'Train Player'}])
    safe = {'person': 'safe', 'match': 'safe_match'}
    unknown = {'person': 'unknown', 'match': 'unmapped_match'}
    assert train_v2.identity_disjoint_ssl([safe, unknown], rows) == [safe]


def test_no_auxiliary_groups_needs_no_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(train_v2, 'DATA', tmp_path)
    assert train_v2.identity_disjoint_ssl([], []) == []
