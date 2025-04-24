<a name="readme-top"></a>

[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Gmail][gmail-shield]][gmail-url]

<br />
<div align="center">

<h1 align="center">CSM</h1>
<h3 align="center">Computational Solid Mechanics</h3>
<h3 align="center">Scientific and Data-Intensive Computing Master Degree</h3>
<h3 align="center">University of Trieste (UniTS) & SISSA</h3>
<h3 align="center">2024-2025</h3>

<p align="center">
    Small library for the solution of simple problems in Computational Solid Mechanics.
    <br />
    <br />
</div>

<!-- TABLE OF CONTENTS -->
<div style="width: 360px; text-align: center; border: 2px solid currentColor; padding: 10px 10px 10px 10px; border-radius: 10px; margin: auto;">
  <h4>📑 Table of Contents</h4>
  <ul style="list-style-type: none; padding: 0;">
    <li><a href="#author-info">Author Info</a></li>
    <li><a href="#description">Description</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#license">License</a></li>
  </ul>
</div>
</br>

<!-- AUTHORS' INFO-->
## Author Info

| Name | Surname | Student ID | UniTS email | Personal email | Master course |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Piero | Zappi | SM3600004 | `piero.zappi@studenti.units.it` | `piero.z.2001@gmail.com` | SDIC |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DESCRIPTION -->
## Description

**CSM** is a small **Python** library for the solution of simple problems in *Computational Solid Mechanics*, namely:

- **Springs**
- **Linear Bars**
- **Plane Trusses**

The library is designed to be easy to use and understand: the package allows the user to easily define the problem and quickly solve it.

Some tutorials are provided in the [**tutorials**](tutorials/) folder, where the user can find examples of how to use the library and understand the tipical workflow of solving a problem using the **CSM** framework.

The library is designed to be extensible, so that the user can easily add new features and functionalities.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DEPENDENCIES -->
## Dependencies

The **CSM** library requires the following dependencies:

- `numpy`
- `matplotlib`

To install the dependencies, you can use the following command in your terminal:

```bash
pip install numpy
pip install matplotlib
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- INSTALLATION -->
## Installation

To install the **CSM** library and being able to import it in your Python scripts a `setup.py` file is provided.\
After downloading the repository, the installation can be done by running the following commands from the root folder of the project in your terminal:

```bash
# Build the module
python3 setup.py build_ext --inplace

# Install the module
python3 -m pip install .
```

After the installation, you can import the library in your Python scripts using the following command:

```python
from CSM import *
```

The library is structured in a modular way, so that you can import only the modules you need.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Contacts -->
[license-shield]: https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge
[license-url]: https://github.com/PieroZ01/Computational_Solid_Mechanics_SDIC/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white&colorB=0077B5
[linkedin-url]: https://www.linkedin.com/in/pierozappi/
[gmail-shield]: https://img.shields.io/badge/-Gmail-red?style=for-the-badge&logo=gmail&logoColor=white&colorB=red
[gmail-url]: mailto:piero.z.2001@gmail.com
