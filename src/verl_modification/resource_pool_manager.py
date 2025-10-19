import ray
from ray.util.placement_group import placement_group
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, RayResourcePool

# A class to enable multiple processes on a single gpu
class CustomRayResourcePool(RayResourcePool):

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        if self.pgs is not None:
            return self.pgs

        pg_name_prefix = name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        # print(f"pg_name_prefix = {pg_name_prefix}")
        if device_name == "npu":
            device_name = "NPU"
        elif device_name == "cuda":
            device_name = "GPU"

        # bundle = {"CPU": self.max_colocate_count}
        bundle = {"CPU": 1} # MODIFIED: colocate_count > 1, but actually 1 WorkerGroup is assigned.
        if self.use_gpu:
            bundle[device_name] = 1 / self.max_colocate_count # MODIFIED: colocate_count > 1, but actually 1 WorkerGroup is assigned.
            if self.accelerator_type is not None:
                bundle[self.accelerator_type] = 1e-4
        pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

        lifetime = "detached" if self.detached else None
        
        pgs = [placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime) for idx, bundles in enumerate(pg_scheme)]

        ray.get([pg.ready() for pg in pgs])

        self.pgs = pgs
        return pgs

class CustomResourcePoolManager(ResourcePoolManager):

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = CustomRayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1.5, name_prefix=resource_pool_name) # Modified max_colocate_count to 2
            self.resource_pool_dict[resource_pool_name] = resource_pool
