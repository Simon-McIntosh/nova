# Web applications

This repository contains Bokeh web applications to facilitate user interaction with Nova's equilibrium generation and 
reconstruction tools.

All web applications require a development version of the Nova codebase installed via poetry from the Nova project's root directory.

```
poetry install
```

To run an application outside a poetry shell, use the following command:

``` 
poetry run bokeh serve --show apps/<appname>
```

To run an application within a poetry shell:

```
poetry shell 
bokeh serve --show apps/<appname> 
```

To develop an application with code changes reflected on the browser side following a page reload:
``` 
poetry shell 
bokeh serve apps/<appname> --dev
``` 
. 
