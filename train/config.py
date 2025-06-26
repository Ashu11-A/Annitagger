import json
from dataclasses import dataclass
from typing import ClassVar, List, Dict, Tuple, Callable, Any

with open('animeName.json', 'r', encoding='utf-8') as file:
    series = json.load(file)

@dataclass
class Config:
  qualities: ClassVar[List[str]] = [
    "144p", "240p", "360p", "480p", "576p", "720p", "1080p", "1440p", "4K", "8K",
    "HD", "Full HD", "Ultra HD", "CAM", "TS", "DVDRip", "BRRip", "WEB-DL", "HDRip", "HDTV"
  ]
  encodes: ClassVar[List[str]] = [
    "AV1", "HEVC", "VP9", "VP8", "x264", "x265", "DivX", "XviD", "MPEG-4", "MPEG-2",
    "H.264", "H.265"
  ]
  file_types: ClassVar[List[str]] = [
    "mkv", "mp4", "avi", "mov", "wmv", "flv", "mpeg", "mpg", "m4v", "webm", "ts", "3gp",
    "vob", "rmvb", "ogg", "ogv", "m2ts"
  ]
  
  audio_types: ClassVar[List[str]] = ["AAC", "AC3", "DTS", "MP3", "FLAC", "Vorbis", "AAC 2.0"]
  sources: ClassVar[List[str]] = ["CR", "Crunchyroll", "WEB", "FUNimation", "NHK", "Aniplus"]
  subs: ClassVar[List[str]] = ["MULTi", "English Dub", "PT-BR", "ENG", "SPA", "JPN"]

  components: ClassVar[Dict[str, Any]] = {
    "submitters": ["Breeze", "Yameii", "Tsundere-Raws", "Honey Lemon Soda", "Kitsune"],
    "series": series,
    "qualities": qualities,
    "encodes": encodes,
    "subs": subs,
    "sources": sources,
    "audio": audio_types,
    "file_types": file_types
  }

  sections: ClassVar[List[Tuple[str, List[str]]]] = [
    ('beginning', ['submitter', 'serie', 'season_episode', 'year']),
    ('middle', ['quality', 'encode', 'source', 'audio']),
    ('end', ['subtitle', 'hash'])
  ]

  component_map: ClassVar[Dict[str, Tuple[Any, str]]] = {
    "submitter": ("submitters", "submitter"),
    "serie": (None, "serie"),
    "year": (None, "year"),
    "quality": ("qualities", "quality"),
    "encode": ("encodes", "encode"),
    "source": ("sources", "source"),
    "audio": ("audio", "audio"),
    "subtitle": ("subs", "subtitle"),
    "hash": (None, "hash"),
    "season_episode": (None, "season_episode"),
  }

  probabilities: ClassVar[Dict[str, Dict[str, float]]] = {
    "submitter": {"beginning": 0.9, "middle": 0.2, "end": 0.1},
    "serie": {"beginning": 1.0, "middle": 0.0, "end": 0.0},
    "year": {"beginning": 0.2, "middle": 0.1, "end": 0.05},
    "quality": {"beginning": 0.3, "middle": 0.9, "end": 0.4},
    "encode": {"beginning": 0.2, "middle": 0.8, "end": 0.3},
    "source": {"beginning": 0.1, "middle": 0.7, "end": 0.3},
    "audio": {"beginning": 0.1, "middle": 0.6, "end": 0.2},
    "subtitle": {"beginning": 0.0, "middle": 0.4, "end": 0.7},
    "hash": {"beginning": 0.0, "middle": 0.1, "end": 0.4},
    "season_episode": {"beginning": 1.0, "middle": 0.0, "end": 0.0},
  }
  
  formats: List[Callable[[str], str]] = [
    lambda t: f"[{t}] ",
    lambda t: f"({t}) ",
    lambda t: f"{t.replace(' ', '.')}.",
    lambda t: f"{t.replace(' ', '-')}-",
    lambda t: f"{t.replace(' ', '_')}_",
  ]
  formats_weights: List[float] = [0.2] * 5
  delimiters: List[str] = ['[', ']', '(', ')', '.', '-', '_']

