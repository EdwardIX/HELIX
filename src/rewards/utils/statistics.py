import os
import ray
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any, Optional

@ray.remote
class RewardMetricsHistActor:
    def __init__(self, plot_kwargs: Optional[Dict[str, Any]] = {}):
        """
        Initialize reward metrics visualizer
        """
        self.plot_kwargs = plot_kwargs
        self.epoch = 0

    def process_reward_metrics(self,
                              data_sources: List[str],
                              sample_inputs: List[str], 
                              infos_dict: Dict[str, List[Any]],
                              proj_base_path: str) -> None:
        """
        Process reward metrics and generate histogram visualization
        
        Args:
            data_sources: List of data source identifiers for each sample
            sample_inputs: List of input prompts for each sample
            infos_dict: Dictionary containing sample values for each variable
        """
        self.epoch += 1
        # 1. Group data by data source and variable
        source_var_values = defaultdict(lambda: defaultdict(list))
        
        # Iterate through all samples
        for i, source in enumerate(data_sources):
            prompt = sample_inputs[i]
            for var_name, values in infos_dict.items():
                # Only process numeric data
                if isinstance(values[i], (int, float)):
                    source_var_values[source][var_name].append(values[i])
        
        # 2. 
        for source, var_dict in source_var_values.items():
            for var_name, values in var_dict.items():
                self._plot_histogram(
                    values=values,
                    data_source=source,
                    var_name=var_name,
                    num_samples=len(values),
                    base_save_dir=os.path.join(proj_base_path, "reward_histograms"),
                    **self.plot_kwargs,
                )
        
        # 3. 
        if len(source_var_values) > 1:
            var_all_values = defaultdict(list)
            for source, var_dict in source_var_values.items():
                for var_name, values in var_dict.items():
                    var_all_values[var_name].extend(values)

            for var_name, values in var_all_values.items():
                self._plot_histogram(
                    values=values,
                    data_source="all",
                    var_name=var_name,
                    num_samples=len(values),
                    base_save_dir=os.path.join(proj_base_path, "reward_histograms"),
                    **self.plot_kwargs,
                )
    
    def _plot_histogram(self, 
                       values: List[float], 
                       data_source: str, 
                       var_name: str, 
                       num_samples: int,
                       base_save_dir: str,
                       bins = 100) -> None:
        """
        
        
        Args:
            values: 
            data_source: 
            var_name: 
            num_samples: 
        """
        plt.figure(figsize=(10, 6))
        
        # 
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # 
        n, bins, patches = plt.hist(values, bins=bins, alpha=0.7, 
                                   color='skyblue', edgecolor='black')
        
        # 
        stats_text = (f'Samples: {num_samples}\n'
                     f'Mean: {mean_val:.4f}\n'
                     f'Std: {std_val:.4f}\n'
                     f'Min: {min_val:.4f}\n'
                     f'Max: {max_val:.4f}')
        plt.gca().text(0.95, 0.95, stats_text, 
                      transform=plt.gca().transAxes,
                      verticalalignment='top', 
                      horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {mean_val:.4f}')
        
        # 
        title_parts = [
            f"Distribution of '{var_name}'",
            f"Data Source: {data_source}",
            f"Samples: {num_samples}"
        ]
        
        if self.epoch is not None:
            title_parts.append(f"Epoch: {self.epoch}")
            
        plt.title('\n'.join(title_parts))
        plt.xlabel(var_name)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 
        filename_base = f"{var_name}"
        if self.epoch is not None:
            filename_base = f"{filename_base}_epoch{self.epoch}"
        
        # 
        filename_base = filename_base.replace(' ', '_').replace('/', '-')
        os.makedirs(os.path.join(base_save_dir, data_source), exist_ok=True)
        save_path = os.path.join(base_save_dir, data_source, filename_base + '.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram to: {save_path}")

        # 
        data_dict = {
            "metadata": {
                "data_source": data_source,
                "variable_name": var_name,
                "epoch": self.epoch,
                "samples_count": num_samples,
                "statistics": {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "min": float(min_val),
                    "max": float(max_val)
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "raw_values": values
        }

        # 
        # 1JSON
        json_path = os.path.join(base_save_dir, data_source, filename_base + '.json')
        with open(json_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
        print(f"Saved raw data to: {json_path}")