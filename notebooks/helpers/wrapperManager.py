import serviceWrapperAsync as servWrapper
import pandas as pd


WRAPPER_REGISTRY = {
    "genderize": servWrapper.GenderizeWrapper,
    "genderapi_io": servWrapper.GenderAPI_IO_Wrapper,
    "genderapi_com": servWrapper.GenderAPI_com_Wrapper,
    "namsor": servWrapper.NamSorWrapper,
    "nameapi": servWrapper.NameAPIWrapper,
}

class WrapperManager():
    def __init__(self, datasource:pd.DataFrame):
        self.datasource = datasource
        self.wrappers = {name: cls(datasource) for name, cls in WRAPPER_REGISTRY.items()}


    async def run_all(self, config: dict) -> dict:
        """
        Run all wrappers according to config.
            
        config = {
            "genderize": {"useLocalization": True},
            "genderapi_io": {"useLocalization": True, "useAI": True},
            "genderapi_com": {"useFullName": True, "useLocalization": False},
            "namsor": {"endpoint": NamSorEndpoint.FULL_NAME},
            "nameapi": {"useFullName": False}
        }
        """
        results = {}
        for name, wrapper in self.wrappers.items():
            kwargs = config.get(name, {})  # get per-wrapper kwargs
            results[name] = await wrapper.get_prediction_async(**kwargs)
        return results
        

    async def run_subset(self, selected: list[str], config: dict = None) -> dict:
        """
        Run only a subset of wrappers, based on their registry names.
        
        Example:
        await manager.run_subset(
            ["genderize", "namsor"],
            config={"genderize": {"useLocalization": True}, "namsor": {"endpoint": NamSorEndpoint.FULL_NAME}}
        )
        """
        config = config or {}
        results = {}
        for name in selected:
            if name not in self.wrappers:
                raise ValueError(f"Unknown wrapper: {name}")
            wrapper = self.wrappers[name]
            kwargs = config.get(name, {})
            results[name] = await wrapper.get_prediction_async(**kwargs)
        return results


