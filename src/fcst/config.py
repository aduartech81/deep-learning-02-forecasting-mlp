"""Carga de configuración por perfil (variable de entorno FC_PROFILE: full | smoke)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    profile: str
    raw: dict

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)

    def path(self, key: str) -> Path:
        p = ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_csv(self) -> Path:
        p = ROOT / self.raw["data"]["raw_csv"]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_config(profile: str | None = None) -> Config:
    profile = profile or os.environ.get("FC_PROFILE", "full")
    with open(ROOT / "config.yaml", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    if profile not in cfg["profiles"]:
        raise ValueError(f"Perfil desconocido: {profile}")
    return Config(profile, cfg["profiles"][profile])
