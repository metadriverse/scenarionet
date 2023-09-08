import pkgutil
import importlib
import argparse

desc = "List all available operations"


def list_modules(package):
    ret = [name for _, name, _ in pkgutil.iter_modules(package.__path__)]
    ret.remove("builder")
    ret.remove("converter")
    ret.remove("verifier")
    ret.remove("common_utils")
    return ret


if __name__ == '__main__':
    import scenarionet

    # exclude current path
    # print(d)
    parser = argparse.ArgumentParser(description=desc)
    parser.parse_args()
    modules = list_modules(scenarionet)
    print("\nAvailable operations (usage python -m scenarionet.operation): \n")
    for module in modules:
        # module="convert_nuplan"
        print(
            "{}scenarionet.{}:   {} \n".format(
                " " * 5, module,
                importlib.import_module("scenarionet.{}".format(module)).desc
            )
        )
