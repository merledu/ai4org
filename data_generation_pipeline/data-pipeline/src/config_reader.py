import yaml
import os

def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    # Construct the full path relative to the script location
    # script_dir = os.path.dirname(os.path.abspath(file))
    # file_path = os.path.join(script_dir, config_path)
    file_path = config_path
    
    try:
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        print(f"Error: The file {config_path} was not found at {file_path}")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None