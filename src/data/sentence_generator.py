"""
Sentence generator for categorized training and test data.

Generates sentences across three categories (animals, music, food) using
template-based randomization to ensure variety and avoid duplicates.
"""

import random
from typing import Dict, List, Set
from src.core.config import Config


class SentenceGenerator:
    """
    Generate categorized sentences for ML training and testing.

    Uses template-based generation with randomization to create diverse
    sentences across three categories. Ensures training and test sets
    are completely disjoint.

    Attributes:
        config: Configuration object
        used_sentences: Set of generated sentences to avoid duplicates

    Example:
        >>> gen = SentenceGenerator(config)
        >>> train = gen.generate_training_set(10)
        >>> test = gen.generate_test_set(10)
        >>> len(train['animals'])
        10
    """

    def __init__(self, config: Config):
        """Initialize generator with configuration."""
        self.config = config
        self.used_sentences: Set[str] = set()  # Track to avoid duplicates
        random.seed(config.random_state)  # For reproducibility

    def generate_training_set(self, n_per_category: int) -> Dict[str, List[str]]:
        """
        Generate training sentences for all categories.

        Args:
            n_per_category: Number of sentences per category

        Returns:
            Dictionary with keys: 'animals', 'music', 'food'
            Each value is a list of sentences

        Example:
            >>> train = gen.generate_training_set(10)
            >>> set(train.keys())
            {'animals', 'music', 'food'}
        """
        self.used_sentences.clear()  # Reset for new generation

        return {
            "animals": self._generate_category("animals", n_per_category),
            "music": self._generate_category("music", n_per_category),
            "food": self._generate_category("food", n_per_category),
        }

    def generate_test_set(self, n_per_category: int) -> Dict[str, List[str]]:
        """
        Generate test sentences (different from training).

        Args:
            n_per_category: Number of sentences per category

        Returns:
            Dictionary with same structure as training set
        """
        return {
            "animals": self._generate_category("animals", n_per_category),
            "music": self._generate_category("music", n_per_category),
            "food": self._generate_category("food", n_per_category),
        }

    def _generate_category(self, category: str, count: int) -> List[str]:
        """Generate sentences for specific category."""
        if category == "animals":
            return self._generate_animal_sentences(count)
        elif category == "music":
            return self._generate_music_sentences(count)
        elif category == "food":
            return self._generate_food_sentences(count)
        else:
            raise ValueError(f"Unknown category: {category}")

    def _generate_animal_sentences(self, count: int) -> List[str]:
        """Generate animal-related sentences."""
        animals = [
            "elephant",
            "leopard",
            "dolphin",
            "hummingbird",
            "eagle",
            "wolf",
            "bear",
            "tiger",
            "lion",
            "zebra",
        ]
        verbs = [
            "stalks",
            "runs",
            "swims",
            "flies",
            "hunts",
            "roams",
            "leaps",
            "climbs",
        ]
        adjectives = ["powerful", "graceful", "swift", "majestic", "wild", "fierce"]
        locations = ["forest", "savanna", "ocean", "mountain", "jungle", "plains"]

        templates = [
            "The {animal} {verb} through the {location} at dawn",
            "A {adj} {animal} {verb} near the river",
            "{animal}s are known for their {adj} movements",
            "The {adj} {animal} is native to the {location}",
        ]

        return self._generate_from_templates(
            templates,
            count,
            animal=animals,
            verb=verbs,
            adj=adjectives,
            location=locations,
        )

    def _generate_music_sentences(self, count: int) -> List[str]:
        """Generate music-related sentences."""
        instruments = [
            "piano",
            "guitar",
            "violin",
            "saxophone",
            "drums",
            "flute",
            "trumpet",
            "cello",
        ]
        genres = ["jazz", "classical", "rock", "blues", "folk", "electronic"]
        adjectives = [
            "melodic",
            "rhythmic",
            "harmonic",
            "soulful",
            "vibrant",
            "powerful",
        ]
        actions = ["performed", "composed", "improvised", "recorded", "played"]

        templates = [
            "The {instrument} produced a {adj} sound",
            "{genre} music features {adj} rhythms and melodies",
            "Musicians {action} the piece with great skill",
            "The {adj} {instrument} solo captivated the audience",
        ]

        return self._generate_from_templates(
            templates,
            count,
            instrument=instruments,
            genre=genres,
            adj=adjectives,
            action=actions,
        )

    def _generate_food_sentences(self, count: int) -> List[str]:
        """Generate food-related sentences."""
        foods = [
            "pasta",
            "bread",
            "rice",
            "cheese",
            "chocolate",
            "fruit",
            "vegetables",
            "seafood",
        ]
        preparations = ["roasted", "grilled", "steamed", "baked", "fried", "sauteed"]
        adjectives = [
            "fresh",
            "delicious",
            "savory",
            "aromatic",
            "organic",
            "flavorful",
        ]
        cuisines = ["Italian", "French", "Asian", "Mediterranean", "Mexican", "Indian"]

        templates = [
            "{adj} {food} tastes wonderful when {prep}",
            "The chef prepared {adj} {food} for dinner",
            "{cuisine} cuisine is famous for its {adj} {food}",
            "{prep} {food} is a staple in many cultures",
        ]

        return self._generate_from_templates(
            templates,
            count,
            food=foods,
            prep=preparations,
            adj=adjectives,
            cuisine=cuisines,
        )

    def _generate_from_templates(
        self, templates: List[str], count: int, **word_lists
    ) -> List[str]:
        """
        Generate sentences from templates with random word substitution.

        Args:
            templates: List of template strings with {placeholders}
            count: Number of sentences to generate
            **word_lists: Named lists of words for substitution

        Returns:
            List of generated sentences
        """
        sentences = []
        max_attempts = count * 10  # Prevent infinite loop
        attempts = 0

        while len(sentences) < count and attempts < max_attempts:
            # Randomly select a template
            template = random.choice(templates)

            # Randomly fill in placeholders
            words = {key: random.choice(values) for key, values in word_lists.items()}

            try:
                sentence = template.format(**words)

                # Check if unique and valid length
                if (
                    sentence not in self.used_sentences
                    and 5 <= len(sentence.split()) <= 20
                ):
                    sentences.append(sentence)
                    self.used_sentences.add(sentence)

            except KeyError:
                # Template placeholder doesn't match word_lists
                pass

            attempts += 1

        if len(sentences) < count:
            raise RuntimeError(
                f"Could not generate {count} unique sentences. "
                f"Only generated {len(sentences)}."
            )

        return sentences
