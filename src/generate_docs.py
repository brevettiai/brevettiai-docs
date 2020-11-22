import os
import pandas as pd
import numpy as np

style_dummy = """<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>"""


def split_file_contents(output_buffer, regex_pattern=".*#.*API:(.*)"):
    content = pd.DataFrame(data={"lines": output_buffer.split("\n")})
    content["section_head"] = content.lines.str.extract(regex_pattern)
    section_head_pos = content.index[~content["section_head"].isna()].values.tolist() + [1000000]
    return [(content.loc[init_pos, "section_head"], content["lines"][init_pos:section_head_pos[index+1]])
            for index, init_pos in enumerate(section_head_pos[:-1])]


def batch_convert(code_dir="src", generated_dir = "generated",
                  convert_cmd=r"python -m jupyter nbconvert --to markdown --execute --ExecutePreprocessor.kernel_name=python {notebook} --output-dir {output}"):

    get_output_dir = lambda path: os.path.join(generated_dir, *path.split(os.path.sep)[1:-1])

    summary_doc = pd.DataFrame(data={"lines": open(os.path.join("docs", "SUMMARY_template.md"), "r").read().split("\n")})
    tutorial_links = []
    for root, dir, files in os.walk(code_dir):
        for file_name in files:
            if os.path.splitext(file_name)[1] == ".ipynb":
                notebook = os.path.join(root, file_name)

                os.system(convert_cmd.format(notebook=notebook, output=get_output_dir(notebook)))
                output_file = os.path.join(get_output_dir(notebook), os.path.basename(notebook).replace("ipynb", "md"))
                output_file_key = '/'.join(output_file.split(os.path.sep))
                output_buffer = open(output_file, "r").read().replace(style_dummy, "")


                output_parts = split_file_contents(output_buffer)

                summary_line = summary_doc.lines.str.extract(fr"\[(.*)\].*{output_file_key}").dropna()
                new_lines = []
                for ii, out_part in enumerate(output_parts):
                    out_name = output_file.replace(".md", f"-{ii}.md")
                    open(out_name, "w").write("\n".join(["# "+out_part[0], *out_part[1].values.tolist()[1:]]))
                    new_lines.append("  " + summary_doc.loc[summary_line.index[0]][0].replace(summary_line.values[0,0], out_part[0]).replace(output_file_key, output_file_key.replace(".md", f"-{ii}.md")))
                summary_doc = pd.concat((summary_doc[:summary_line.index[0]+1], pd.DataFrame(data={"lines": new_lines}), summary_doc[summary_line.index[0]+1:]))

                colab_link = f"[{os.path.basename(notebook)}](https://githubtocolab.com/criterion-ai/brevettiai-docs/blob/master/{'/'.join(notebook.split(os.path.sep))})"
                tutorial_links.append(colab_link)
                new_intro = output_buffer[:np.where(np.array(list(output_buffer))=="#")[0][1]]
                outro = f"""To explore the code by examples, please run the in the notebook that can be found on colab on this link {colab_link}"""
                open(output_file, "w").write(new_intro + outro)
    open(os.path.join("generated", "SUMMARY.md"), "w").write("\n".join(summary_doc.lines.values.tolist()))

    tutorial_buffer = open(os.path.join("..", "docs", "developers", "tutorials", "tutorial_template.md"), "r").read()
    with open(os.path.join("..", generated_dir, "developers", "tutorials", "tutorials.md"), "w") as tut_file:
        tut_file.write(tutorial_buffer + "\n".join([] + [f"* {link}" for link in tutorial_links]))


if __name__ == "__main__":
    batch_convert()
