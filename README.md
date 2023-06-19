# Plant Reaction-Diffusion Simulation

This project aims to implement various Reaction-Diffusion models to simulate the effects of harsh climates on plant communities. It is inspired by Problem A of the Mathematical Competition in Modeling from the year 2023. This project implements various models using the Julia Programming Language, which was chosen for its superior performance in scientific computation relative to Python.

## Background

Different species of plants respond differently to stresses, such as drought. The frequency and severity of droughts can vary, and it has been observed that plant communities with a greater number of species are better adapted to drought conditions over successive generations. This project seeks to explore the relationship between the number of species in a plant community and their ability to adapt to irregular weather cycles, specifically focusing on drought periods.

### Requirements

The projects requirements were, as outlined by the Mathematical Competition in Modeling, are as follows:

1. Develop a mathematical model to predict how a plant community changes over time when exposed to irregular weather cycles, including drought periods. The model should consider interactions between different species during droughts.
2. Draw conclusions from the model regarding the long-term interactions between plant communities and the larger environment. Consider the impact of the number and types of species, frequency and variation of drought occurrence, as well as factors like pollution and habitat reduction.
3. Provide recommendations based on the model to ensure the long-term viability of plant communities and assess their impact on the larger environment.

## File Structure

The project repository has the following structure:

- `bib/`: Contains BibTex bibliography files.
- `docs/`: Contains in-depth discussion on the implementations found in source code.
- `imgs/`: Contains various images created with Plots.jl throughout various simulations.
- `src/`: Contains the source code files for the project.
- `test/`: Contains various tests written to check for code errors.
- `LICENSE`: Outlines the GNU Affero General Public License v3.0.
- `README.md`: This file.

## Goals

Current goals are:

- [x] Implement the basic algorithms which form the backbone of the simulation.
- [ ] Implement a basic running simulation of this project.
- [ ] Implement a more sophisticated simulation using Poisson Random Processes to simulate variation in weather and climate.
- [ ] Develop more sophisticated models which account for root structure, ground topology and multiple differing forms of biomass.
- [ ] Implement a more sophisticated simulation which takes into account root structure, ground topology, runoff, and other factors found in nature.

## Usage

To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/plant-reaction-diffusion.git`
1. Run the simulation script: `julia simulation_name.jl`

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's repository.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

---

This README provides an overview of the project, its purpose, and the requirements outlined in the problem statement. It also highlights the structure of the project solution, how to use it, and how to contribute. Remember to update the sections with relevant information based on your implementation.

Please note that the README file is a brief summary, and additional documentation may be required within the project itself or in the report accompanying it.

Best of luck with your plant simulation project! If you have any further questions, feel free to ask.
