import json
import sys

def run_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as nb_file:
        notebook = json.load(nb_file)
    
    # Collect all code cell sources into a single string.
    full_code = ""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            # Each cell's source is a list of strings. We join them.
            cell_code = "".join(cell.get("source", []))
            # Optionally, add a separator (e.g., a comment) between cells.
            full_code += f"\n# Begin cell\n{cell_code}\n# End cell\n"
    return full_code

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <notebook.ipynb>")
        sys.exit(1)
    
    notebook_file = sys.argv[1]
    code_to_run = run_notebook(notebook_file)
    # Execute the combined notebook code under __main__ context.
    exec(code_to_run, {"__name__": "__main__"})
