from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Type


class ReviewerProtocol(ABC):
    @abstractmethod
    def review_paper(self, pdf_file: bytes) -> Any:
        pass


class ReviewerRegistry:
    _reviewers: Dict[str, Type[ReviewerProtocol]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[ReviewerProtocol]], Type[ReviewerProtocol]]:
        def decorator(reviewer_class: Type[ReviewerProtocol]) -> Type[ReviewerProtocol]:
            cls._reviewers[name] = reviewer_class
            return reviewer_class

        return decorator

    @classmethod
    def create_reviewer(cls, name: str, config: dict = None) -> ReviewerProtocol:
        if name not in cls._reviewers:
            available = ", ".join(cls._reviewers.keys())
            raise ValueError(f"Unknown reviewer '{name}'. Available: {available}")

        reviewer_class = cls._reviewers[name]
        if config is not None:
            return reviewer_class(config_override=config)
        else:
            return reviewer_class()

    @classmethod
    def list_reviewers(cls) -> list[str]:
        return list(cls._reviewers.keys())

    @classmethod
    def get_reviewer_class(cls, name: str) -> Type[ReviewerProtocol]:
        if name not in cls._reviewers:
            raise ValueError(f"Unknown reviewer: {name}")
        return cls._reviewers[name]
