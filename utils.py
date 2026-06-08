import json
from logger import debug_msg, DebugLevel
import os
import time


def get_duration_string(start_time, end_time):
    duration = end_time - start_time
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_string = ""
    duration_string += " %sd" % days if days > 0 else ""
    duration_string += " %sh" % hours if hours > 0 else ""
    duration_string += " %sm" % minutes if minutes > 0 else ""
    duration_string += " %ss" % seconds if seconds > 0 or duration_string.strip() == "" else ""
    return duration_string.strip()


def load_json(path_to_file):
    with open(path_to_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_to_file(filename, json_content):
    with open(filename, "w") as outfile:
        json.dump(json_content, outfile, ensure_ascii=False, indent=4)


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return


def get_clean_model_name(model):
    return model.replace(".", "_").replace(":", "-")


def check_prompt_already_processed(prompt_id, repetition, all_responses):
    try:
        return len(all_responses[str(prompt_id)]["responses"]) >= repetition
    except:
        return False


def load_all_responses_from_file(language, model, results_directory, results_filename):
    path_to_file = os.path.join(results_directory,results_filename.format(
        language=language,
        model=get_clean_model_name(model)
    ))
    try:
        if os.path.exists(path_to_file):
            with open(path_to_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return {}


def save_all_responses_to_json_file(responses, language, model, results_directory, results_filename):
    path_to_file = os.path.join(results_directory, results_filename.format(
        language=language,
        model=get_clean_model_name(model)
    ))
    save_json_to_file(path_to_file, responses)


def save_response_to_temp_file(prompt_id, response, language, model, results_directory):
    timestamp = str(time.time())
    filepath = os.path.join(results_directory, f"_temp_{language}_{get_clean_model_name(model)}_{timestamp}.json")
    json_content = {"prompt_id": str(prompt_id),
                    "language": language,
                    "model": model,
                    "response": response}
    save_json_to_file(filepath, json_content)


def read_and_delete_temp_files(language, model, results_directory):
    all_temp_data = []
    for filename in os.listdir(results_directory):
        if filename.startswith(f"_temp_{language}_{get_clean_model_name(model)}_"):
            try:
                all_temp_data.append(load_json(os.path.join(results_directory, filename)))
            except:
                debug_msg(f"Invalid JSON in {os.path.join(results_directory, filename)}", DebugLevel.WARNING)
            os.remove(os.path.join(results_directory, filename))
    return all_temp_data


def add_temp_data(all_responses, all_temp_data, language, model):
    for temp_data in all_temp_data:
        assert str(temp_data["prompt_id"]) in all_responses
        assert temp_data["language"] == language
        assert temp_data["model"] == model
        all_responses[str(temp_data["prompt_id"])]["responses"].append(temp_data["response"])
    return all_responses
