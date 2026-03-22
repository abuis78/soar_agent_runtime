"""
soar_agent_runtime_view.py
Custom widget context builder for SOAR Agent Runtime.
"""
import json


def display_run_agent(provides, all_app_runs, context):
    context["results"] = []
    for summary, action_results in all_app_runs:
        for result in action_results:
            if result.get_status() == "success":
                data = result.get_data()
                if data:
                    d = data[0]
                    try:
                        step_log = json.loads(d.get("step_log", "[]"))
                    except Exception:
                        step_log = []
                    context["results"].append({
                        "data": d,
                        "step_log": step_log,
                        "summary": result.get_summary(),
                        "param": result.get_param(),
                    })
    return "/widgets/soar_agent_runtime_view.html"
