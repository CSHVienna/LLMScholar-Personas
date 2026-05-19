from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelSpec:
    """Declarative configuration for a sensitivity analysis."""
    persona_vars: List[str]
    context_vars: List[str]
    metric: str = None
    llm_id: str = 'model'
    llm_attrs: List[str] = field(default_factory=list)  # used by Model 2 only
    structural_vars: List[str] = field(default_factory=list)  # used for nested missingness handling
    
    def __post_init__(self):
        # Auto-exclude metric from its own predictors
        if self.persona_vars is not None:
            self.persona_vars = [v for v in self.persona_vars if v != self.metric]
        
        if self.context_vars is not None:
            self.context_vars = [v for v in self.context_vars if v != self.metric]
        
        if self.structural_vars is not None:
            self.structural_vars = [v for v in self.structural_vars if v != self.metric]

    @property
    def prompt_vars(self) -> list[str]:
        """Manipulated prompt variables. Used for plotting and interpretation."""
        return self.persona_vars + self.context_vars

    @property
    def all_predictors(self) -> list[str]:
        """All predictors entering the model formula (prompt + structural)."""
        return self.prompt_vars + self.structural_vars
    
    def var_kind(self, term: str) -> str:
        """Tag a coefficient term as social / technical / llm_attr / interaction."""
        if term == 'Intercept':
            return 'intercept'
        if ':' in term:
            return 'interaction'
        if any(v in term for v in self.persona_vars):
            return 'persona'
        if any(v in term for v in self.context_vars):
            return 'context'
        if self.llm_id in term:
            return 'llm_fe'
        if any(a in term for a in self.llm_attrs):
            return 'llm_attr'
        if any(sv in term for sv in self.structural_vars):
            return 'structural'
        return 'other'