"""
Bootstrap sampling for initial training data selection.

This module provides tools to select a diverse, representative sample
of activities for initial annotation - maximizing coverage of different
activity types while minimizing redundancy.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import logging

logger = logging.getLogger(__name__)


# Singleton instance
_bootstrap_sampler_instance = None


def get_bootstrap_sampler() -> 'BootstrapSampler':
    """Get or create the singleton BootstrapSampler instance."""
    global _bootstrap_sampler_instance
    if _bootstrap_sampler_instance is None:
        _bootstrap_sampler_instance = BootstrapSampler()
    return _bootstrap_sampler_instance


class BootstrapSampler:
    """
    High-level interface for bootstrap sampling.
    
    Wraps DiverseSampler with convenient methods for the API routes.
    """
    
    def __init__(self):
        """Initialize the bootstrap sampler."""
        self.sampler = DiverseSampler(
            n_clusters=50,
            min_similarity_threshold=0.80,
            random_state=42
        )
    
    def select_diverse_sample(
        self,
        df: pd.DataFrame,
        n_samples: int = 200,
        text_column: str = 'ACTIVITY_DESCRIPTION'
    ) -> pd.DataFrame:
        """
        Select a diverse sample of activities.
        
        Args:
            df: DataFrame with activities
            n_samples: Number of samples to select
            text_column: Column containing text descriptions
            
        Returns:
            DataFrame with selected diverse samples
        """
        sampled_df, metadata = self.sampler.sample_diverse(
            df=df,
            text_column=text_column,
            n_samples=n_samples
        )
        logger.info(f"Selected {len(sampled_df)} diverse samples using {metadata.get('method', 'unknown')} method")
        return sampled_df
    
    def export_for_annotation(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'json'
    ) -> dict:
        """
        Export samples in specified format for annotation.
        
        Args:
            df: DataFrame to export
            output_path: Path to save the file
            format: 'json', 'csv', or 'markdown'
            
        Returns:
            Metadata about the export
        """
        import json
        import os
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if format == 'json':
            return self._export_json(df, output_path)
        elif format == 'csv':
            return self._export_csv(df, output_path)
        elif format == 'markdown':
            return self._export_markdown(df, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_json(self, df: pd.DataFrame, output_path: str) -> dict:
        """Export as JSON for easy import/programmatic annotation."""
        import json
        
        # Convert to records, handling NaN values
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    record[col] = None
                else:
                    record[col] = val
            # Add empty label fields
            record['PHASE_LABEL'] = None
            record['DISCIPLINE_LABEL'] = None
            record['WORK_TYPE_LABEL'] = None
            record['LOCATION_LABEL'] = None
            record['NOTES'] = None
            records.append(record)
        
        export_data = {
            'metadata': {
                'n_samples': len(df),
                'columns': df.columns.tolist(),
                'label_fields': ['PHASE_LABEL', 'DISCIPLINE_LABEL', 'WORK_TYPE_LABEL', 'LOCATION_LABEL', 'NOTES']
            },
            'samples': records
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        return {'format': 'json', 'path': output_path, 'n_samples': len(df)}
    
    def _export_csv(self, df: pd.DataFrame, output_path: str) -> dict:
        """Export as CSV with label columns."""
        export_df = df.copy()
        
        # Add empty label columns
        for col in ['PHASE_LABEL', 'DISCIPLINE_LABEL', 'WORK_TYPE_LABEL', 'LOCATION_LABEL', 'NOTES']:
            if col not in export_df.columns:
                export_df[col] = ''
        
        export_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        return {'format': 'csv', 'path': output_path, 'n_samples': len(df)}
    
    def _export_markdown(self, df: pd.DataFrame, output_path: str) -> dict:
        """Export as Markdown for LLM annotation."""
        lines = [
            "# Bootstrap Sample for Annotation",
            "",
            "Please annotate each activity with the appropriate labels.",
            "",
            "## Label Options",
            "",
            "- **PHASE_LABEL**: Project phase (e.g., Engineering, Procurement, Construction, Commissioning)",
            "- **DISCIPLINE_LABEL**: Technical discipline (e.g., Mechanical, Electrical, Piping, Civil)",
            "- **WORK_TYPE_LABEL**: Type of work (e.g., Design, Review, Installation, Testing)",
            "- **LOCATION_LABEL**: Location type (e.g., Onshore, Offshore)",
            "",
            "---",
            ""
        ]
        
        # Get key columns
        text_col = None
        for col in ['ACTIVITY_DESCRIPTION', 'description', 'SHORT_NAME']:
            if col in df.columns:
                text_col = col
                break
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            lines.append(f"## Activity {i}")
            lines.append("")
            
            # Show key info
            if 'PROJECT_ID' in df.columns:
                lines.append(f"**Project:** {row.get('PROJECT_ID', 'N/A')}")
            if 'SUB_PROJECT_ID' in df.columns:
                lines.append(f"**Sub-project:** {row.get('SUB_PROJECT_ID', 'N/A')}")
            if 'ACTIVITY_NO' in df.columns:
                lines.append(f"**Activity No:** {row.get('ACTIVITY_NO', 'N/A')}")
            
            if text_col:
                lines.append(f"**Description:** {row.get(text_col, 'N/A')}")
            
            lines.append("")
            lines.append("**Labels to assign:**")
            lines.append("- PHASE_LABEL: ")
            lines.append("- DISCIPLINE_LABEL: ")
            lines.append("- WORK_TYPE_LABEL: ")
            lines.append("- LOCATION_LABEL: ")
            lines.append("- NOTES: ")
            lines.append("")
            lines.append("---")
            lines.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return {'format': 'markdown', 'path': output_path, 'n_samples': len(df)}


class DiverseSampler:
    """
    Selects a diverse sample of activities using clustering and similarity metrics.
    
    Strategy:
    1. Convert activity descriptions to TF-IDF vectors
    2. Cluster activities into groups
    3. Sample from each cluster proportionally
    4. Within clusters, select most representative (centroid-closest) samples
    5. Ensure minimum diversity between selected samples
    """
    
    def __init__(
        self,
        n_clusters: int = 50,
        min_similarity_threshold: float = 0.85,
        random_state: int = 42
    ):
        """
        Initialize the diverse sampler.
        
        Args:
            n_clusters: Number of clusters to create (more = finer grouping)
            min_similarity_threshold: Maximum allowed similarity between samples
                                     (lower = more diverse)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.min_similarity_threshold = min_similarity_threshold
        self.random_state = random_state
        self.vectorizer = None
        self.vectors = None
        
    def sample_diverse(
        self,
        df: pd.DataFrame,
        text_column: str,
        n_samples: int = 200,
        id_column: Optional[str] = None,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Select a diverse sample of activities.
        
        Args:
            df: DataFrame with activities
            text_column: Column containing text descriptions
            n_samples: Number of samples to select
            id_column: Optional ID column (defaults to first column)
            stratify_column: Optional column to stratify by (e.g., discipline_code)
            
        Returns:
            Tuple of (sampled DataFrame, metadata dict)
        """
        if len(df) <= n_samples:
            return df.copy(), {'method': 'full', 'reason': 'Dataset smaller than sample size'}
        
        # Prepare text data
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Filter out empty texts
        valid_mask = [len(t.strip()) > 5 for t in texts]
        valid_indices = [i for i, v in enumerate(valid_mask) if v]
        valid_texts = [texts[i] for i in valid_indices]
        
        if len(valid_texts) < n_samples:
            # Not enough valid texts, return random sample
            return df.sample(n=min(n_samples, len(df)), random_state=self.random_state), {
                'method': 'random',
                'reason': 'Not enough valid text descriptions'
            }
        
        print(f"Vectorizing {len(valid_texts)} activities...")
        
        # Vectorize texts using TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.vectors = self.vectorizer.fit_transform(valid_texts)
        
        # Adjust cluster count based on data size
        actual_clusters = min(self.n_clusters, len(valid_texts) // 4, n_samples // 2)
        actual_clusters = max(actual_clusters, 10)
        
        print(f"Clustering into {actual_clusters} groups...")
        
        # Cluster the activities
        if len(valid_texts) > 10000:
            # Use MiniBatch for large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=actual_clusters,
                random_state=self.random_state,
                batch_size=1000
            )
        else:
            kmeans = KMeans(
                n_clusters=actual_clusters,
                random_state=self.random_state,
                n_init=10
            )
        
        cluster_labels = kmeans.fit_predict(self.vectors)
        
        # Calculate samples per cluster (proportional to cluster size)
        cluster_sizes = np.bincount(cluster_labels)
        samples_per_cluster = np.maximum(
            1,
            (cluster_sizes / cluster_sizes.sum() * n_samples).astype(int)
        )
        
        # Adjust to hit exact target
        while samples_per_cluster.sum() < n_samples:
            # Add to largest clusters
            idx = np.argmax(cluster_sizes - samples_per_cluster)
            samples_per_cluster[idx] += 1
        while samples_per_cluster.sum() > n_samples:
            # Remove from smallest contributions
            idx = np.argmax(samples_per_cluster - 1)
            if samples_per_cluster[idx] > 1:
                samples_per_cluster[idx] -= 1
        
        print(f"Selecting diverse samples from each cluster...")
        
        # Select samples from each cluster
        selected_indices = []
        
        for cluster_id in range(actual_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            n_from_cluster = min(samples_per_cluster[cluster_id], len(cluster_indices))
            
            if n_from_cluster == 0:
                continue
            
            # Get vectors for this cluster
            cluster_vectors = self.vectors[cluster_indices]
            
            # Calculate distance to centroid
            centroid = cluster_vectors.mean(axis=0)
            if hasattr(centroid, 'A'):
                centroid = centroid.A  # Convert sparse matrix
            
            distances = []
            for i, idx in enumerate(cluster_indices):
                vec = cluster_vectors[i]
                if hasattr(vec, 'toarray'):
                    vec = vec.toarray().flatten()
                dist = np.linalg.norm(vec - centroid.flatten())
                distances.append((idx, dist))
            
            # Sort by distance to centroid (closest first = most representative)
            distances.sort(key=lambda x: x[1])
            
            # Select samples, ensuring diversity within cluster
            cluster_selected = []
            for idx, _ in distances:
                if len(cluster_selected) >= n_from_cluster:
                    break
                
                # Check similarity with already selected
                if len(cluster_selected) > 0 and len(selected_indices) > 0:
                    candidate_vec = self.vectors[idx]
                    max_sim = 0
                    for sel_idx in cluster_selected[-5:]:  # Check last 5 for efficiency
                        sim = cosine_similarity(candidate_vec, self.vectors[sel_idx])[0, 0]
                        max_sim = max(max_sim, sim)
                    
                    if max_sim > self.min_similarity_threshold:
                        continue  # Skip too-similar samples
                
                cluster_selected.append(idx)
            
            selected_indices.extend(cluster_selected)
        
        # Map back to original DataFrame indices
        original_indices = [valid_indices[i] for i in selected_indices]
        
        # Ensure we have exactly n_samples (or close to it)
        if len(original_indices) < n_samples:
            # Fill remaining with random samples not yet selected
            remaining = set(valid_indices) - set(original_indices)
            additional = random.sample(list(remaining), min(n_samples - len(original_indices), len(remaining)))
            original_indices.extend(additional)
        
        sampled_df = df.iloc[original_indices].copy()
        
        # Add metadata
        metadata = {
            'method': 'diverse_clustering',
            'n_clusters': actual_clusters,
            'n_samples': len(sampled_df),
            'min_similarity_threshold': self.min_similarity_threshold,
            'cluster_distribution': {
                int(k): int(v) for k, v in zip(
                    range(actual_clusters),
                    np.bincount(cluster_labels[selected_indices], minlength=actual_clusters)
                )
            }
        }
        
        return sampled_df, metadata
    
    def export_for_annotation(
        self,
        df: pd.DataFrame,
        output_path: str,
        text_column: str = 'ACTIVITY_DESCRIPTION',
        n_samples: int = 200,
        include_columns: Optional[List[str]] = None
    ) -> dict:
        """
        Export a diverse sample for manual annotation.
        
        Creates a CSV file with selected activities and empty label columns
        for manual annotation.
        
        Args:
            df: Full DataFrame
            output_path: Path to save the CSV
            text_column: Column with activity descriptions
            n_samples: Number of samples to export
            include_columns: Additional columns to include (auto-detects if None)
            
        Returns:
            Metadata about the export
        """
        # Find the text column (case-insensitive)
        text_col = None
        for col in df.columns:
            if col.upper() == text_column.upper():
                text_col = col
                break
        
        if text_col is None:
            # Try alternatives
            for alt in ['DESCRIPTION', 'SHORT_NAME', 'NAME']:
                for col in df.columns:
                    if col.upper() == alt:
                        text_col = col
                        break
                if text_col:
                    break
        
        if text_col is None:
            raise ValueError(f"Could not find text column. Available: {df.columns.tolist()}")
        
        # Sample diverse activities
        sampled_df, metadata = self.sample_diverse(df, text_col, n_samples)
        
        # Determine columns to include
        if include_columns is None:
            # Auto-detect useful columns
            include_columns = []
            column_map = {col.upper(): col for col in df.columns}
            
            for col_name in [
                'ACTIVITY_SEQ', 'ACTIVITY_DESCRIPTION', 'SHORT_NAME',
                'PROJECT_ID', 'SUB_PROJECT_ID', 'DISCIPLINE_CODE',
                'PROJECT_NAME', 'SUB_PROJECT_DESCRIPTION', 'EARLY_START',
                'PLAN_HRS', 'PROGRESS'
            ]:
                if col_name in column_map:
                    include_columns.append(column_map[col_name])
        
        # Filter to available columns
        include_columns = [c for c in include_columns if c in sampled_df.columns]
        
        # Create export DataFrame
        export_df = sampled_df[include_columns].copy()
        
        # Add empty label columns for annotation
        export_df['PHASE_LABEL'] = ''
        export_df['DISCIPLINE_LABEL'] = ''
        export_df['WORK_TYPE_LABEL'] = ''
        export_df['NOTES'] = ''
        
        # Save to CSV
        export_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        metadata['output_path'] = output_path
        metadata['columns_exported'] = include_columns
        metadata['label_columns'] = ['PHASE_LABEL', 'DISCIPLINE_LABEL', 'WORK_TYPE_LABEL', 'NOTES']
        
        print(f"\n✓ Exported {len(export_df)} diverse activities to: {output_path}")
        print(f"  Columns: {', '.join(include_columns[:5])}...")
        print(f"  Label columns to fill: {metadata['label_columns']}")
        
        return metadata


def export_bootstrap_sample(
    input_csv: str = 'data/training_data.csv',
    output_csv: str = 'data/bootstrap_sample_200.csv',
    n_samples: int = 200
) -> dict:
    """
    Convenience function to export a bootstrap sample from cached data.
    
    Args:
        input_csv: Path to full training data
        output_csv: Path to save bootstrap sample
        n_samples: Number of samples
        
    Returns:
        Export metadata
    """
    import os
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Training data not found at {input_csv}. Load data from database first.")
    
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} activities from {input_csv}")
    
    sampler = DiverseSampler(
        n_clusters=50,
        min_similarity_threshold=0.80,
        random_state=42
    )
    
    metadata = sampler.export_for_annotation(
        df,
        output_csv,
        n_samples=n_samples
    )
    
    return metadata


if __name__ == '__main__':
    # Run from command line
    import sys
    
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    output = sys.argv[2] if len(sys.argv) > 2 else f'data/bootstrap_sample_{n}.csv'
    
    metadata = export_bootstrap_sample(n_samples=n, output_csv=output)
    print(f"\nMetadata: {metadata}")
