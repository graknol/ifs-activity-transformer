"""
Label mapping utilities for activity classification.

This module provides utilities to derive multi-label classifications from
activity data using the discipline_codes.json configuration.

The mapper transforms raw IFS activity data into structured labels for:
- Phase (ENGINEERING, FABRICATION, INSTALLATION_OFFSHORE, etc.)
- Discipline (ELECTRICAL, PIPING, STRUCTURAL, etc.)
- Work Type (DIRECT, INDIRECT, VENDORS_3RD_PARTY, etc.)
- Location (ONSHORE, OFFSHORE)
- Contract Type (RB, LS)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from app.config import DisciplineConfig, Config


class LabelMapper:
    """
    Maps activity data to multi-label classifications.
    
    Uses discipline codes extracted from sub_project_id to derive:
    - Phase labels (from first letter of discipline code)
    - Technical discipline labels
    - Work type labels
    - Location labels
    
    Example:
        >>> mapper = LabelMapper()
        >>> labels = mapper.get_labels_from_code('QL')
        >>> # Returns: {'phase': 'INSTALLATION_OFFSHORE', 'discipline': 'PIPING', 
        >>> #          'work_type': 'DIRECT', 'location': 'OFFSHORE'}
    """
    
    def __init__(self, discipline_config: Optional[DisciplineConfig] = None):
        """
        Initialize label mapper with discipline configuration.
        
        Args:
            discipline_config: DisciplineConfig instance. Loads default if None.
        """
        self.config = discipline_config or DisciplineConfig.load()
        
        # Build label indices for efficient encoding
        self._phase_to_idx = {label: i for i, label in enumerate(self.config.phase_labels)}
        self._discipline_to_idx = {label: i for i, label in enumerate(self.config.discipline_labels)}
        self._work_type_to_idx = {label: i for i, label in enumerate(self.config.work_type_labels)}
        self._location_to_idx = {label: i for i, label in enumerate(self.config.location_labels)}
        
        # Combined label info
        self._all_labels = self.config.all_labels
        self._label_to_idx = {label: i for i, label in enumerate(self._all_labels)}
        
    @property
    def num_labels(self) -> int:
        """Total number of label categories."""
        return len(self._all_labels)
    
    @property
    def label_names(self) -> List[str]:
        """List of all label names in order."""
        return self._all_labels.copy()
    
    @property
    def num_phase_labels(self) -> int:
        """Number of phase labels."""
        return len(self.config.phase_labels)
    
    @property
    def num_discipline_labels(self) -> int:
        """Number of discipline labels."""
        return len(self.config.discipline_labels)
    
    @property
    def num_work_type_labels(self) -> int:
        """Number of work type labels."""
        return len(self.config.work_type_labels)
    
    @property
    def num_location_labels(self) -> int:
        """Number of location labels."""
        return len(self.config.location_labels)
    
    def get_labels_from_code(self, discipline_code: str) -> Dict[str, Optional[str]]:
        """
        Get label dictionary for a discipline code.
        
        Args:
            discipline_code: Two-letter discipline code (e.g., 'KE', 'QL')
            
        Returns:
            Dict with keys: phase, discipline, work_type, location (values may be None)
        """
        return self.config.get_labels_for_code(discipline_code)
    
    def extract_discipline_code(self, sub_project_id: str) -> Optional[str]:
        """
        Extract discipline code from sub_project_id.
        
        In IFS, the discipline code is typically the first 2 characters
        of the sub_project_id (e.g., 'KE001' -> 'KE').
        
        Args:
            sub_project_id: Full sub-project ID string
            
        Returns:
            Two-letter discipline code or None
        """
        if sub_project_id and len(sub_project_id) >= 2:
            code = sub_project_id[:2].upper()
            # Verify it's a valid discipline code
            if self.config.get_code_description(code):
                return code
        return None
    
    def encode_labels_binary(self, labels_dict: Dict[str, Optional[str]]) -> np.ndarray:
        """
        Encode label dictionary to binary vector.
        
        Args:
            labels_dict: Dict with phase, discipline, work_type, location keys
            
        Returns:
            Binary numpy array of shape (num_labels,)
        """
        binary = np.zeros(self.num_labels, dtype=np.float32)
        
        # Phase label
        phase = labels_dict.get('phase')
        if phase and phase in self._label_to_idx:
            binary[self._label_to_idx[phase]] = 1.0
        
        # Discipline label
        discipline = labels_dict.get('discipline')
        if discipline and discipline in self._label_to_idx:
            binary[self._label_to_idx[discipline]] = 1.0
        
        # Work type label
        work_type = labels_dict.get('work_type')
        if work_type and work_type in self._label_to_idx:
            binary[self._label_to_idx[work_type]] = 1.0
        
        # Location label
        location = labels_dict.get('location')
        if location and location in self._label_to_idx:
            binary[self._label_to_idx[location]] = 1.0
        
        return binary
    
    def encode_from_discipline_code(self, discipline_code: str) -> np.ndarray:
        """
        Encode discipline code directly to binary label vector.
        
        Args:
            discipline_code: Two-letter discipline code
            
        Returns:
            Binary numpy array of shape (num_labels,)
        """
        labels = self.get_labels_from_code(discipline_code)
        return self.encode_labels_binary(labels)
    
    def decode_binary_labels(
        self, 
        binary: np.ndarray, 
        threshold: float = 0.5
    ) -> Dict[str, List[str]]:
        """
        Decode binary vector back to label names.
        
        Args:
            binary: Binary array of predictions
            threshold: Threshold for positive prediction
            
        Returns:
            Dict with lists of predicted labels by category
        """
        predictions = {
            'phase': [],
            'discipline': [],
            'work_type': [],
            'location': []
        }
        
        # Determine offsets for each category
        phase_end = len(self.config.phase_labels)
        discipline_end = phase_end + len(self.config.discipline_labels)
        work_type_end = discipline_end + len(self.config.work_type_labels)
        
        for i, (label, value) in enumerate(zip(self._all_labels, binary)):
            if value >= threshold:
                if i < phase_end:
                    predictions['phase'].append(label)
                elif i < discipline_end:
                    predictions['discipline'].append(label)
                elif i < work_type_end:
                    predictions['work_type'].append(label)
                else:
                    predictions['location'].append(label)
        
        return predictions
    
    def prepare_training_dataframe(
        self,
        df: pd.DataFrame,
        sub_project_id_column: str = 'sub_project_id',
        text_column: str = 'activity_description'
    ) -> pd.DataFrame:
        """
        Prepare DataFrame with derived labels for training.
        
        Adds columns for each label category based on discipline codes
        extracted from sub_project_id.
        
        Args:
            df: Input DataFrame with activity data
            sub_project_id_column: Column containing sub-project IDs
            text_column: Column containing text descriptions
            
        Returns:
            DataFrame with added label columns
        """
        result = df.copy()
        
        # Extract discipline codes
        result['discipline_code'] = result[sub_project_id_column].apply(
            self.extract_discipline_code
        )
        
        # Get labels for each code
        def get_label_value(code, label_key):
            if code:
                labels = self.get_labels_from_code(code)
                return labels.get(label_key)
            return None
        
        # Add individual label columns
        result['label_phase'] = result['discipline_code'].apply(
            lambda x: get_label_value(x, 'phase')
        )
        result['label_discipline'] = result['discipline_code'].apply(
            lambda x: get_label_value(x, 'discipline')
        )
        result['label_work_type'] = result['discipline_code'].apply(
            lambda x: get_label_value(x, 'work_type')
        )
        result['label_location'] = result['discipline_code'].apply(
            lambda x: get_label_value(x, 'location')
        )
        
        # Add binary encoded labels as a column of arrays
        result['binary_labels'] = result['discipline_code'].apply(
            lambda x: self.encode_from_discipline_code(x) if x else np.zeros(self.num_labels)
        )
        
        return result
    
    def get_label_columns(self) -> List[str]:
        """Get names of all binary label columns for model output."""
        return [f'label_{name.lower()}' for name in self._all_labels]
    
    def create_binary_label_matrix(
        self,
        df: pd.DataFrame,
        discipline_code_column: str = 'discipline_code'
    ) -> np.ndarray:
        """
        Create binary label matrix for entire DataFrame.
        
        Args:
            df: DataFrame with discipline_code column
            discipline_code_column: Column containing discipline codes
            
        Returns:
            Binary matrix of shape (n_samples, num_labels)
        """
        labels = np.zeros((len(df), self.num_labels), dtype=np.float32)
        
        for i, code in enumerate(df[discipline_code_column]):
            if code:
                labels[i] = self.encode_from_discipline_code(code)
        
        return labels
    
    def get_sub_project_info(self, sub_project_id: str) -> Dict[str, Any]:
        """
        Get full information about a sub-project ID from NEW system structure.
        
        Args:
            sub_project_id: Sub-project ID (e.g., '3000', '7500')
            
        Returns:
            Dict with name, contract_type, phase, etc.
        """
        sub_projects = self.config.target_sub_project_structure.get(
            'order_projects', {}
        ).get('sub_projects', {})
        
        info = sub_projects.get(sub_project_id, {})
        
        # Add derived information
        info['derived_phase'] = self.config.get_phase_from_sub_project_id(sub_project_id)
        info['derived_contract_type'] = self.config.get_contract_type_from_sub_project_id(sub_project_id)
        
        return info
    
    def get_phase_code_mapping(self) -> Dict[str, str]:
        """
        Get mapping of phase prefix letters to phase names.
        
        Returns:
            Dict like {'K': 'Engineering', 'Q': 'Installation Offshore', ...}
        """
        phases = self.config.project_main_levels.get('phases', {})
        return {
            data['prefix']: data['name']
            for data in phases.values()
            if 'prefix' in data
        }
    
    def validate_dataset(self, df: pd.DataFrame, sub_project_id_column: str = 'sub_project_id') -> Dict[str, Any]:
        """
        Validate dataset and report coverage statistics.
        
        Args:
            df: DataFrame to validate
            sub_project_id_column: Column containing sub-project IDs
            
        Returns:
            Dict with validation statistics
        """
        result = {
            'total_records': len(df),
            'valid_codes': 0,
            'invalid_codes': 0,
            'missing_codes': 0,
            'code_distribution': {},
            'phase_distribution': {},
            'unmapped_codes': set()
        }
        
        for sub_proj_id in df[sub_project_id_column]:
            if not sub_proj_id:
                result['missing_codes'] += 1
                continue
                
            code = self.extract_discipline_code(sub_proj_id)
            if code:
                result['valid_codes'] += 1
                result['code_distribution'][code] = result['code_distribution'].get(code, 0) + 1
                
                labels = self.get_labels_from_code(code)
                phase = labels.get('phase')
                if phase:
                    result['phase_distribution'][phase] = result['phase_distribution'].get(phase, 0) + 1
            else:
                result['invalid_codes'] += 1
                if len(sub_proj_id) >= 2:
                    result['unmapped_codes'].add(sub_proj_id[:2])
        
        result['unmapped_codes'] = list(result['unmapped_codes'])
        result['coverage_pct'] = (result['valid_codes'] / result['total_records'] * 100) if result['total_records'] > 0 else 0
        
        return result


def get_label_mapper() -> LabelMapper:
    """
    Factory function to get a configured LabelMapper instance.
    
    Returns:
        LabelMapper instance with default configuration
    """
    return LabelMapper()
