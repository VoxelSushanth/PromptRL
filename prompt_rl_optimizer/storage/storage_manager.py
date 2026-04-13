"""
Storage and Logging Module.

Provides persistent storage for:
- Prompts and variants
- LLM outputs
- Evaluation metrics
- Optimization history

Supports SQLite, JSON, and in-memory storage.
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import threading

from core.models import (
    PromptVariant, LLMOutput, EvaluationMetrics, 
    Trial, OptimizationState, ExperimentResult
)
from configs.experiment_config import StorageConfig


class BaseStorage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_trial(self, trial: Trial) -> None:
        """Save a single trial."""
        pass
    
    @abstractmethod
    def save_trials(self, trials: List[Trial]) -> None:
        """Save multiple trials."""
        pass
    
    @abstractmethod
    def get_trials(self, experiment_id: str) -> List[Trial]:
        """Retrieve all trials for an experiment."""
        pass
    
    @abstractmethod
    def get_best_prompt(self, experiment_id: str) -> Optional[PromptVariant]:
        """Get the best performing prompt."""
        pass


class MemoryStorage(BaseStorage):
    """In-memory storage for testing and prototyping."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.trials: Dict[str, List[Trial]] = {}
        self.prompts: Dict[str, PromptVariant] = {}
        self.outputs: Dict[str, LLMOutput] = {}
        self.metrics: Dict[str, EvaluationMetrics] = {}
        self.experiments: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def save_trial(self, trial: Trial) -> None:
        with self._lock:
            exp_id = trial.id.split('_')[0] if '_' in trial.id else 'default'
            
            if exp_id not in self.trials:
                self.trials[exp_id] = []
            
            self.trials[exp_id].append(trial)
            self.prompts[trial.prompt.id] = trial.prompt
            self.outputs[trial.output.id] = trial.output
            self.metrics[trial.metrics.output_id] = trial.metrics
    
    def save_trials(self, trials: List[Trial]) -> None:
        for trial in trials:
            self.save_trial(trial)
    
    def get_trials(self, experiment_id: str) -> List[Trial]:
        return self.trials.get(experiment_id, [])
    
    def get_best_prompt(self, experiment_id: str) -> Optional[PromptVariant]:
        trials = self.get_trials(experiment_id)
        
        if not trials:
            return None
        
        # Find trial with highest reward
        best_trial = max(trials, key=lambda t: t.metrics.final_reward)
        return best_trial.prompt
    
    def get_statistics(self, experiment_id: str) -> Dict:
        """Get statistics for an experiment."""
        trials = self.get_trials(experiment_id)
        
        if not trials:
            return {'count': 0}
        
        rewards = [t.metrics.final_reward for t in trials]
        
        return {
            'count': len(trials),
            'mean_reward': sum(rewards) / len(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards)
        }
    
    def clear(self, experiment_id: str = None) -> None:
        """Clear stored data."""
        with self._lock:
            if experiment_id:
                self.trials.pop(experiment_id, None)
            else:
                self.trials.clear()
                self.prompts.clear()
                self.outputs.clear()
                self.metrics.clear()


class SQLiteStorage(BaseStorage):
    """SQLite-based persistent storage."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.db_path = Path(config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Trials table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    iteration INTEGER,
                    prompt_id TEXT,
                    output_id TEXT,
                    task_description TEXT,
                    test_input TEXT,
                    expected_output TEXT,
                    created_at TIMESTAMP
                )
            ''')
            
            # Prompts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    base_prompt TEXT,
                    variant_prompt TEXT,
                    generation_strategy TEXT,
                    metadata TEXT,
                    total_trials INTEGER,
                    cumulative_reward REAL,
                    average_reward REAL,
                    created_at TIMESTAMP
                )
            ''')
            
            # Outputs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS outputs (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT,
                    prompt_text TEXT,
                    input_data TEXT,
                    output_text TEXT,
                    model_name TEXT,
                    tokens_used INTEGER,
                    latency_ms REAL,
                    error_message TEXT,
                    retry_count INTEGER,
                    created_at TIMESTAMP
                )
            ''')
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    output_id TEXT PRIMARY KEY,
                    correctness_score REAL,
                    format_compliance_score REAL,
                    completeness_score REAL,
                    conciseness_score REAL,
                    output_length INTEGER,
                    num_sections INTEGER,
                    has_required_format BOOLEAN,
                    repetition_ratio REAL,
                    judge_feedback TEXT,
                    rule_based_errors TEXT,
                    final_reward REAL,
                    evaluated_at TIMESTAMP
                )
            ''')
            
            # Experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    config TEXT,
                    best_prompt_id TEXT,
                    best_reward REAL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trials_experiment ON trials(experiment_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trials_iteration ON trials(iteration)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_reward ON metrics(final_reward)')
            
            conn.commit()
            conn.close()
    
    def save_trial(self, trial: Trial) -> None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                # Save trial
                cursor.execute('''
                    INSERT OR REPLACE INTO trials 
                    (id, experiment_id, iteration, prompt_id, output_id, 
                     task_description, test_input, expected_output, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trial.id,
                    trial.id.split('_')[0] if '_' in trial.id else 'default',
                    trial.iteration,
                    trial.prompt.id,
                    trial.output.id,
                    trial.task_description,
                    json.dumps(trial.test_input),
                    json.dumps(trial.expected_output) if trial.expected_output else None,
                    trial.created_at.isoformat()
                ))
                
                # Save prompt
                cursor.execute('''
                    INSERT OR REPLACE INTO prompts
                    (id, base_prompt, variant_prompt, generation_strategy, metadata,
                     total_trials, cumulative_reward, average_reward, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trial.prompt.id,
                    trial.prompt.base_prompt,
                    trial.prompt.variant_prompt,
                    trial.prompt.generation_strategy,
                    json.dumps(trial.prompt.metadata),
                    trial.prompt.total_trials,
                    trial.prompt.cumulative_reward,
                    trial.prompt.average_reward,
                    trial.prompt.created_at.isoformat()
                ))
                
                # Save output
                cursor.execute('''
                    INSERT OR REPLACE INTO outputs
                    (id, prompt_id, prompt_text, input_data, output_text,
                     model_name, tokens_used, latency_ms, error_message, retry_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trial.output.id,
                    trial.output.prompt_id,
                    trial.output.prompt_text,
                    json.dumps(trial.output.input_data),
                    trial.output.output_text,
                    trial.output.model_name,
                    trial.output.tokens_used,
                    trial.output.latency_ms,
                    trial.output.error_message,
                    trial.output.retry_count,
                    trial.output.created_at.isoformat()
                ))
                
                # Save metrics
                cursor.execute('''
                    INSERT OR REPLACE INTO metrics
                    (output_id, correctness_score, format_compliance_score,
                     completeness_score, conciseness_score, output_length,
                     num_sections, has_required_format, repetition_ratio,
                     judge_feedback, rule_based_errors, final_reward, evaluated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trial.metrics.output_id,
                    trial.metrics.correctness_score,
                    trial.metrics.format_compliance_score,
                    trial.metrics.completeness_score,
                    trial.metrics.conciseness_score,
                    trial.metrics.output_length,
                    trial.metrics.num_sections,
                    trial.metrics.has_required_format,
                    trial.metrics.repetition_ratio,
                    trial.metrics.judge_feedback,
                    json.dumps(trial.metrics.rule_based_errors),
                    trial.metrics.final_reward,
                    trial.metrics.evaluated_at.isoformat()
                ))
                
                conn.commit()
            
            finally:
                conn.close()
    
    def save_trials(self, trials: List[Trial]) -> None:
        for trial in trials:
            self.save_trial(trial)
    
    def get_trials(self, experiment_id: str) -> List[Trial]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM trials WHERE experiment_id = ?', (experiment_id,))
            rows = cursor.fetchall()
            
            conn.close()
            
            trials = []
            for row in rows:
                # Reconstruct Trial object from row
                # This is simplified; full implementation would join with other tables
                pass
            
            return trials
    
    def get_best_prompt(self, experiment_id: str) -> Optional[PromptVariant]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT p.* FROM prompts p
                JOIN trials t ON p.id = t.prompt_id
                JOIN metrics m ON t.output_id = m.output_id
                WHERE t.experiment_id = ?
                ORDER BY m.final_reward DESC
                LIMIT 1
            ''', (experiment_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return PromptVariant(
                    id=row[0],
                    base_prompt=row[1],
                    variant_prompt=row[2],
                    generation_strategy=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                    total_trials=row[5],
                    cumulative_reward=row[6],
                    average_reward=row[7]
                )
            
            return None
    
    def checkpoint(self, state: OptimizationState) -> None:
        """Save optimization checkpoint."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO experiments
                (id, name, config, best_prompt_id, best_reward, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                state.experiment_id,
                state.experiment_id,
                json.dumps({}),
                state.best_prompt_id,
                state.best_reward,
                state.started_at.isoformat(),
                None
            ))
            
            conn.commit()
            conn.close()


class JSONStorage(BaseStorage):
    """JSON file-based storage for logging and export."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.log_dir = Path(config.log_directory)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self.buffer: List[Trial] = []
        self.buffer_size = 10
    
    def save_trial(self, trial: Trial) -> None:
        with self._lock:
            self.buffer.append(trial)
            
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def save_trials(self, trials: List[Trial]) -> None:
        with self._lock:
            self.buffer.extend(trials)
            
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Flush buffer to disk."""
        if not self.buffer:
            return
        
        # Group by experiment
        experiments = {}
        for trial in self.buffer:
            exp_id = trial.id.split('_')[0] if '_' in trial.id else 'default'
            if exp_id not in experiments:
                experiments[exp_id] = []
            experiments[exp_id].append(trial)
        
        # Write to files
        for exp_id, trials in experiments.items():
            filename = self.log_dir / f"{exp_id}_trials.json"
            
            # Load existing data
            existing_data = []
            if filename.exists():
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            
            # Append new trials
            for trial in trials:
                existing_data.append(trial.model_dump())
            
            # Write back
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
        
        self.buffer = []
    
    def get_trials(self, experiment_id: str) -> List[Trial]:
        self._flush_buffer()
        
        filename = self.log_dir / f"{experiment_id}_trials.json"
        
        if not filename.exists():
            return []
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert dicts back to Trial objects
        # Simplified; full implementation would handle nested objects
        return []
    
    def get_best_prompt(self, experiment_id: str) -> Optional[PromptVariant]:
        trials = self.get_trials(experiment_id)
        
        if not trials:
            return None
        
        best_trial = max(trials, key=lambda t: t.metrics.final_reward)
        return best_trial.prompt
    
    def export_results(self, result: ExperimentResult) -> None:
        """Export experiment results to JSON."""
        filename = self.log_dir / f"{result.experiment_id}_results.json"
        
        with open(filename, 'w') as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
    
    def flush(self) -> None:
        """Force flush buffer to disk."""
        self._flush_buffer()


class StorageManager:
    """
    Manages multiple storage backends.
    
    Provides unified interface for:
    - Primary storage (SQLite or memory)
    - Backup/logging (JSON)
    - Checkpointing
    """
    
    def __init__(self, config: StorageConfig):
        self.config = config
        
        # Initialize primary storage
        if config.storage_type == "sqlite":
            self.primary = SQLiteStorage(config)
        elif config.storage_type == "json":
            self.primary = JSONStorage(config)
        else:
            self.primary = MemoryStorage(config)
        
        # Always keep JSON log for export
        self.logger = JSONStorage(config)
    
    def save_trial(self, trial: Trial) -> None:
        """Save trial to both primary and logger."""
        self.primary.save_trial(trial)
        self.logger.save_trial(trial)
    
    def save_trials(self, trials: List[Trial]) -> None:
        """Save multiple trials."""
        self.primary.save_trials(trials)
        self.logger.save_trials(trials)
    
    def get_trials(self, experiment_id: str) -> List[Trial]:
        """Retrieve trials from primary storage."""
        return self.primary.get_trials(experiment_id)
    
    def get_best_prompt(self, experiment_id: str) -> Optional[PromptVariant]:
        """Get best prompt from primary storage."""
        return self.primary.get_best_prompt(experiment_id)
    
    def checkpoint(self, state: OptimizationState) -> None:
        """Save optimization checkpoint."""
        if hasattr(self.primary, 'checkpoint'):
            self.primary.checkpoint(state)
    
    def export_results(self, result: ExperimentResult) -> None:
        """Export final results."""
        self.logger.export_results(result)
        self.logger.flush()
    
    def flush(self) -> None:
        """Flush all buffers."""
        if hasattr(self.logger, 'flush'):
            self.logger.flush()
