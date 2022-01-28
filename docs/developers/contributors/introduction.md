---
The *contributors* section describes how a *data scientist* should use available tools and conform to coding standards to contribute to the API's and tools of the Brevetti AI platform.
---
## Coding guidelines

* Refer to [Google Python Stule Guide](https://google.github.io/styleguide/pyguide.html) 
* Comments: Add comments to make the code readable. **NB: Elaborate!**
* Use PEP-8 to guide you in writing beautiful code. Read and apply the PyCharm tips found here: [Code Quality Assistance Tips and Tricks, or How to Make Your Code Look Pretty?](https://www.jetbrains.com/help/pycharm/tutorial-code-quality-assistance-tips-and-tricks.html)
* Create a serializable training pipeline: derive classes from <code>brevettiai.interfaces.vue_schema_utils.VueSettingsModule</code>

## Brevettiai development environment setup

<p> If you want to develop features on the brevettiai library follow these steps:

Alternatives:

* Install the package directly with pip from the source files
* add the repository directory to PYTHONPATH
* add the code repositories to *source* directories in e.g. PyCharm

### Requirements:

Python poetry: https://python-poetry.org/docs/master/#installing-with-the-official-installer

## *brevettiai* installation steps:

* Pull the sources from the repository git clone git@bitbucket.org:criterionai/core.git

* Create folder for your code and navigate to it

* Add the following content to the pyproject.toml file, alternatively use poetry init to generate it:Update the path of the brevettiai package if your project folder is not adjacent to the core repository. </p>
<pre><code>
[tool.poetry]
name = "development"
version = "0.1.0"
description = ""
authors = [""]

[tool.poetry.dependencies]
python = ">=3.7,<3.11"
brevettiai = {path = "../core", extras = ["cv2", "tf", "tfa"], develop = true}

[tool.poetry.dev-dependencies]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
</code></pre>

* Install additional dependencies / development dependencies with poetry add ...

* Run your virtual environment poetry run or poetry shell

## Brevettiai library releases

The Bitbucket pipeline for the core repository now builds and uploads the library to PyPi.

For the deployment step you must be aware of the following three things:
* Once a version exists on PyPi it cannot be removed. Please test the code well beforehand
* Only one version may exist with the same version number <code>poetry version</code> can help you manage updates. [Poetry version documentation](https://python-poetry.org/docs/cli/#version) 
* When updating dependencies run <code> poetry update </code>  find new versions and then commit the <code>poetry.lock</code>  to apply the changes to the build on the pipeline. <code> poetry update</code>  can also get a package name if you are only looking to change one package version