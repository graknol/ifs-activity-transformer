"""
Text preparation utilities for optimal RoBERTa input.

This module prepares activity data into rich text representations
that leverage all available contextual information for better predictions.

Key strategies:
1. Combine multiple text fields into structured input
2. Use natural language templates that RoBERTa understands
3. Normalize and clean text appropriately
4. Handle Norwegian/English mixed content
5. Include hierarchical context from project structure
"""
import re
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class TextPrepConfig:
    """Configuration for text preparation."""
    
    # Field inclusion
    include_project_context: bool = True
    include_sub_project_context: bool = True
    include_hierarchy_context: bool = True
    include_metadata: bool = True
    
    # Template style
    use_structured_template: bool = True
    use_separator_tokens: bool = True  # Use [SEP] between sections
    
    # Text cleaning
    normalize_whitespace: bool = True
    remove_activity_codes: bool = False  # Keep codes, they're informative
    lowercase: bool = False  # RoBERTa handles casing
    
    # Max lengths (before tokenization)
    max_description_length: int = 300
    max_context_length: int = 200


class ActivityTextPreparer:
    """
    Prepares activity data into optimal text format for RoBERTa.
    
    Instead of just using the activity description, this creates
    rich text representations that include:
    - Activity description (primary)
    - Activity short name/code
    - Sub-project description and hierarchy
    - Project name and context
    - Numerical features as natural language
    
    Example output:
    "Activity: Install electrical cable trays in module B deck 3.
     Task: NE-2450 Cable tray installation.
     Sub-project: Electrical Installation - Onshore.
     Project: Johan Sverdrup Phase 2 - Topside electrical systems.
     Hours: 450 planned, 380 used. Progress: 85%."
    """
    
    def __init__(self, config: Optional[TextPrepConfig] = None):
        """
        Initialize text preparer.
        
        Args:
            config: Text preparation configuration
        """
        self.config = config or TextPrepConfig()
    
    def prepare_single(
        self,
        activity_description: str,
        short_name: Optional[str] = None,
        sub_project_description: Optional[str] = None,
        parent_description: Optional[str] = None,
        project_name: Optional[str] = None,
        project_description: Optional[str] = None,
        plan_hrs: Optional[float] = None,
        used_hrs: Optional[float] = None,
        progress: Optional[float] = None,
        discipline_code: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Prepare a single activity into optimal text format.
        
        Args:
            activity_description: Main activity description (required)
            short_name: Activity short name/code
            sub_project_description: Sub-project description
            parent_description: Parent sub-project description
            project_name: Project name
            project_description: Project description
            plan_hrs: Planned hours
            used_hrs: Used hours
            progress: Progress percentage
            discipline_code: Discipline code (for reference, not used in text)
            **kwargs: Additional fields (ignored)
            
        Returns:
            Formatted text string for RoBERTa input
        """
        parts = []
        
        # 1. Primary: Activity description (most important)
        desc = self._clean_text(activity_description, self.config.max_description_length)
        if desc:
            parts.append(f"Activity: {desc}")
        
        # 2. Activity short name (often contains useful codes)
        if short_name and self.config.include_metadata:
            short = self._clean_text(short_name, 50)
            if short and short.lower() != desc.lower()[:len(short)]:
                parts.append(f"Task: {short}")
        
        # 3. Sub-project context
        if self.config.include_sub_project_context:
            if sub_project_description:
                sp_desc = self._clean_text(sub_project_description, 100)
                if sp_desc:
                    parts.append(f"Sub-project: {sp_desc}")
            
            # Parent sub-project for hierarchy
            if self.config.include_hierarchy_context and parent_description:
                parent = self._clean_text(parent_description, 80)
                if parent:
                    parts.append(f"Parent: {parent}")
        
        # 4. Project context
        if self.config.include_project_context:
            if project_name:
                pname = self._clean_text(project_name, 80)
                if pname:
                    parts.append(f"Project: {pname}")
        
        # 5. Numerical context (as natural language)
        if self.config.include_metadata:
            num_parts = []
            if plan_hrs is not None and plan_hrs > 0:
                num_parts.append(f"{int(plan_hrs)} planned hours")
            if used_hrs is not None and used_hrs > 0:
                num_parts.append(f"{int(used_hrs)} used")
            if progress is not None and progress > 0:
                num_parts.append(f"{int(progress)}% complete")
            
            if num_parts:
                parts.append(f"Status: {', '.join(num_parts)}.")
        
        # Join with appropriate separator
        if self.config.use_separator_tokens:
            # Use [SEP] token that RoBERTa understands
            return " [SEP] ".join(parts)
        else:
            return " ".join(parts)
    
    def prepare_minimal(self, activity_description: str) -> str:
        """
        Prepare minimal text (description only) for comparison.
        
        Args:
            activity_description: Activity description
            
        Returns:
            Cleaned description only
        """
        return self._clean_text(activity_description, self.config.max_description_length)
    
    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        output_column: str = 'prepared_text',
        minimal_column: Optional[str] = 'text_minimal'
    ) -> pd.DataFrame:
        """
        Prepare text for entire DataFrame.
        
        Args:
            df: DataFrame with activity data
            output_column: Name for prepared text column
            minimal_column: Name for minimal text column (None to skip)
            
        Returns:
            DataFrame with added text columns
        """
        result = df.copy()
        
        # Map column names (handle variations)
        col_map = self._detect_columns(df)
        
        # Prepare rich text
        result[output_column] = result.apply(
            lambda row: self.prepare_single(
                activity_description=self._get_value(row, col_map.get('activity_description')),
                short_name=self._get_value(row, col_map.get('short_name')),
                sub_project_description=self._get_value(row, col_map.get('sub_project_description')),
                parent_description=self._get_value(row, col_map.get('parent_description')),
                project_name=self._get_value(row, col_map.get('project_name')),
                project_description=self._get_value(row, col_map.get('project_description')),
                plan_hrs=self._get_value(row, col_map.get('plan_hrs')),
                used_hrs=self._get_value(row, col_map.get('used_hrs')),
                progress=self._get_value(row, col_map.get('progress')),
                discipline_code=self._get_value(row, col_map.get('discipline_code'))
            ),
            axis=1
        )
        
        # Prepare minimal text for comparison
        if minimal_column:
            desc_col = col_map.get('activity_description')
            if desc_col and desc_col in result.columns:
                result[minimal_column] = result[desc_col].apply(
                    lambda x: self.prepare_minimal(x) if pd.notna(x) else ""
                )
        
        return result
    
    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect column names in DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Mapping of standard names to actual column names
        """
        columns = {c.lower(): c for c in df.columns}
        
        mapping = {}
        
        # Activity description
        for name in ['activity_description', 'description', 'desc', 'activity_desc']:
            if name in columns:
                mapping['activity_description'] = columns[name]
                break
        
        # Short name
        for name in ['short_name', 'shortname', 'activity_no', 'task_code']:
            if name in columns:
                mapping['short_name'] = columns[name]
                break
        
        # Sub-project description
        for name in ['sub_project_description', 'subproject_description', 'sp_description']:
            if name in columns:
                mapping['sub_project_description'] = columns[name]
                break
        
        # Parent description
        for name in ['parent_description', 'parent_sub_project_description']:
            if name in columns:
                mapping['parent_description'] = columns[name]
                break
        
        # Project name
        for name in ['project_name', 'proj_name', 'project']:
            if name in columns:
                mapping['project_name'] = columns[name]
                break
        
        # Project description
        for name in ['project_description', 'proj_description']:
            if name in columns:
                mapping['project_description'] = columns[name]
                break
        
        # Numerical fields
        for name in ['plan_hrs', 'planned_hours', 'budgeted_hours']:
            if name in columns:
                mapping['plan_hrs'] = columns[name]
                break
        
        for name in ['used_hrs', 'actual_hours', 'used_hours']:
            if name in columns:
                mapping['used_hrs'] = columns[name]
                break
        
        for name in ['progress', 'completion', 'percent_complete']:
            if name in columns:
                mapping['progress'] = columns[name]
                break
        
        for name in ['discipline_code', 'disc_code']:
            if name in columns:
                mapping['discipline_code'] = columns[name]
                break
        
        return mapping
    
    def _get_value(self, row: pd.Series, column: Optional[str]) -> Any:
        """Get value from row, handling missing columns."""
        if column and column in row.index:
            val = row[column]
            return val if pd.notna(val) else None
        return None
    
    def _clean_text(self, text: str, max_length: int = 500) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            max_length: Maximum length to keep
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove or normalize special characters
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Truncate if needed
        if len(text) > max_length:
            # Try to cut at word boundary
            text = text[:max_length]
            last_space = text.rfind(' ')
            if last_space > max_length * 0.8:
                text = text[:last_space]
            text = text.rstrip() + "..."
        
        return text


class TemplateBasedPreparer(ActivityTextPreparer):
    """
    Alternative preparer using task-specific templates.
    
    Uses templates designed for different prediction tasks:
    - Phase prediction: Focus on work type indicators
    - Discipline prediction: Focus on technical terms
    - Location prediction: Focus on installation context
    """
    
    TEMPLATES = {
        'general': (
            "Classify this oil & gas activity: {description}. "
            "Task code: {short_name}. "
            "Work package: {sub_project_description}. "
            "Project: {project_name}."
        ),
        'phase_focused': (
            "Determine the project phase for: {description}. "
            "This is part of {sub_project_description} "
            "in {project_name}."
        ),
        'discipline_focused': (
            "Identify the engineering discipline for: {description}. "
            "Technical context: {sub_project_description}. "
            "Task: {short_name}."
        ),
        'location_focused': (
            "Determine if this is onshore or offshore work: {description}. "
            "Work package: {sub_project_description}. "
            "Installation context: {project_name}."
        )
    }
    
    def __init__(
        self, 
        template_name: str = 'general',
        config: Optional[TextPrepConfig] = None
    ):
        """
        Initialize template-based preparer.
        
        Args:
            template_name: Name of template to use
            config: Text preparation configuration
        """
        super().__init__(config)
        self.template = self.TEMPLATES.get(template_name, self.TEMPLATES['general'])
    
    def prepare_single(
        self,
        activity_description: str,
        short_name: Optional[str] = None,
        sub_project_description: Optional[str] = None,
        project_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Prepare text using template."""
        return self.template.format(
            description=self._clean_text(activity_description, 300),
            short_name=self._clean_text(short_name or "N/A", 50),
            sub_project_description=self._clean_text(sub_project_description or "unspecified", 100),
            project_name=self._clean_text(project_name or "project", 80)
        )


def create_training_text(
    df: pd.DataFrame,
    strategy: str = 'rich',
    config: Optional[TextPrepConfig] = None
) -> pd.DataFrame:
    """
    Convenience function to prepare training text.
    
    Args:
        df: DataFrame with activity data
        strategy: 'rich' (full context), 'minimal' (description only), 
                  'template' (template-based)
        config: Optional configuration
        
    Returns:
        DataFrame with prepared text columns
    """
    if strategy == 'minimal':
        config = config or TextPrepConfig(
            include_project_context=False,
            include_sub_project_context=False,
            include_hierarchy_context=False,
            include_metadata=False
        )
        preparer = ActivityTextPreparer(config)
    elif strategy == 'template':
        preparer = TemplateBasedPreparer('general', config)
    else:  # 'rich' or default
        preparer = ActivityTextPreparer(config)
    
    return preparer.prepare_dataframe(df)


def compare_text_strategies(df: pd.DataFrame, sample_size: int = 5) -> None:
    """
    Compare different text preparation strategies on sample data.
    
    Args:
        df: DataFrame with activity data
        sample_size: Number of samples to show
    """
    sample = df.head(sample_size)
    
    # Rich preparation
    rich_preparer = ActivityTextPreparer()
    rich_result = rich_preparer.prepare_dataframe(sample)
    
    # Minimal preparation
    minimal_config = TextPrepConfig(
        include_project_context=False,
        include_sub_project_context=False,
        include_hierarchy_context=False,
        include_metadata=False
    )
    minimal_preparer = ActivityTextPreparer(minimal_config)
    minimal_result = minimal_preparer.prepare_dataframe(sample)
    
    # Template preparation
    template_preparer = TemplateBasedPreparer('general')
    template_result = template_preparer.prepare_dataframe(sample)
    
    print("=" * 80)
    print("TEXT PREPARATION STRATEGY COMPARISON")
    print("=" * 80)
    
    for i in range(min(sample_size, len(sample))):
        print(f"\n--- Sample {i+1} ---")
        print(f"\nMINIMAL (description only):")
        print(f"  {minimal_result.iloc[i]['text_minimal'][:200]}...")
        print(f"\nRICH (full context):")
        print(f"  {rich_result.iloc[i]['prepared_text'][:300]}...")
        print(f"\nTEMPLATE (structured):")
        print(f"  {template_result.iloc[i]['prepared_text'][:300]}...")
