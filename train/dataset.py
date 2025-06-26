import random
import json
import string
from config import Config
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

@dataclass
class Entity:
  start: int
  end: int
  label: str

class ComponentGenerator:
  def __init__(self):
    self.position: int = 0
    self.filename: str = ""
    self.entities: List[Entity] = []

  def add_component(self, text: str, entities: List[Dict]) -> 'ComponentGenerator':
    start = self.position
    self.filename += text
    for ent in entities:
      self.entities.append(Entity(
        start=start + ent["offset"],
        end=start + ent["offset"] + ent["length"],
        label=ent["label"]
      ))
    self.position += len(text)
    return self

  def clean_filename(self) -> None:
    original = self.filename
    self.filename = self.filename.rstrip(' ._-')
    trimmed = len(original) - len(self.filename)
    if trimmed > 0:
      self.position -= trimmed
      self.entities = [ent for ent in self.entities if ent.end <= len(self.filename)]

class ComponentFactory:
  def __init__(self):
    self.chosen_format = random.choices(Config.formats, weights=Config.formats_weights, k=1)[0]

  def text_formatter(self, text: str):
    formatted: str = self.chosen_format(text.strip())
    offset, length = 0, len(formatted.strip())
    
    for delim in Config.delimiters:
      if formatted.startswith(delim):
        offset = 1
        length -= 2
        break
      if formatted.endswith(delim):
        length -= 1

    return formatted, offset, length
  
  def generate_serie(self) -> Tuple[str, List[Dict]]:
    value = random.choice(Config.components['series'])
    formatted, offset, length = self.text_formatter(value, chosen_format)
    
    if (formatted.startswith('[') or formatted.startswith('(')):
      formatted = formatted.replace("[", "").replace("]", "").replace("(", "").replace(")", "")

    return formatted, [{"label": "SERIE", "offset": 0, "length": len(formatted)}]
  
  def generate_year(self) -> Tuple[str, List[Dict]]:
    year = str(random.randint(1980, 2100))
    formatted, offset, length = self._text_formatter(year)
    return formatted, [{"label": "YEAR", "offset": offset, "length": length}]
  
  def generate_hash(self) -> Tuple[str, List[Dict]]:
    hash_val = ''.join(random.choices(string.hexdigits, k=8))
    formatted, offset, length = self.text_formatter(hash_val)
    return formatted, [{"label": "HASH", "offset": offset, "length": length}]
  
  def generate_season_episode(self) -> Tuple[str, List[Dict]]:
    season = f"{random.randint(1, 99):02d}"
    episode = f"{random.randint(1, 99):02d}"
    
    if random.random() < 0.5:
      value = f"S{season}E{episode}"
      formatted, offset, _ = self.text_formatter(value)
      entities = [
        {"label": "SEASON", "offset": offset + 1, "length": 2},
        {"label": "EPISODE", "offset": offset + 4, "length": 2}
      ]
    else:
      formatted, offset, _ = self.text_formatter(episode)
      entities = [{"label": "EPISODE", "offset": offset, "length": 2}]
    return formatted, entities
  
  def generate_standard_component(self, component_key: str, label: str) -> Tuple[str, List[Dict]]:
      value = random.choice(self.config.COMPONENTS[component_key])
      formatted, offset, length = self._text_formatter(value)
      return formatted, [{"label": label.upper(), "offset": offset, "length": length}]

  def generate(self, label: str) -> Optional[Tuple[str, List[Dict]]]:
      comp_key, ent_label = self.config.COMPONENT_MAP[label]
      if comp_key:
          return self._generate_standard_component(comp_key, ent_label)
      handler = getattr(self, f"_generate_{label}", None)
      return handler() if handler else None


class DatasetGenerator:
    def __init__(self, config: Config):
        self.config = config

    def _select_components(self) -> List[Tuple[str, List[Dict]]]:
        components = {"beginning": [], "middle": [], "end": []}
        factory = ComponentFactory(self.config)

        for section_name, labels in self.config.SECTIONS:
            for label in labels:
                prob = self.config.PROBABILITIES[label].get(section_name, 0.0)
                if random.random() < prob:
                    generated = factory.generate(label)
                    if generated:
                        components[section_name].append(generated)

        if random.random() < 0.3:
            random.shuffle(components["middle"])
            random.shuffle(components["end"])

        return components["beginning"] + components["middle"] + components["end"]

    def generate_filename(self) -> Dict:
        generator = ComponentGenerator()
        components = self._select_components()

        for text, entities in components:
            generator.add_component(text, entities)

        generator._clean_filename()
        self._add_file_extension(generator)
        generator.filename = generator.filename.strip()
        return {
            "filename": generator.filename,
            "entities": [{"start": e.start, "end": e.end, "label": e.label} for e in generator.entities]
        }

    def _add_file_extension(self, generator: ComponentGenerator) -> None:
        file_type = random.choice(self.config.COMPONENTS["file_types"])
        generator.add_component(f".{file_type}", [
            {"label": "FILE_TYPE", "offset": 1, "length": len(file_type)}
        ])

    def generate_dataset(self, num_samples: int = 1000) -> List[Dict]:
        return [self.generate_filename() for _ in range(num_samples)]


def main():
    with open('animeName.json', 'r', encoding='utf-8') as file:
        series_data = json.load(file)
    
    config = Config(series_data)
    dataset_generator = DatasetGenerator(config)
    dataset = dataset_generator.generate_dataset()
    
    with open("anime_filenames_v2.json", "w", encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()