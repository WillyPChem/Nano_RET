# Nano_RET
Code to simulate Resonance Energy Transfer between a nanoparticle and a small molecule dye (modeled after Malachite Green).
This code solves coupled Liouville-Lindblad equations of motion for the nanoparticle and the dye molecule, where both systems
are coupled to an incident light pulse and to each other through their transition dipole moments.
Please see the document "Nano_RET_User_Guide.pdf" for more information on numerical experiments that can be performed with the code.

# Compiling and Running on a Linux/Unix/Mac OS system from the terminal

- To compile the code, type 
`make`

- To run the code, type
`./Nano_RET.x`
and follow the prompts

- The following data are generated and written to files in the DATA/ folder for subsequent analysis

-- The absorption and scattering cross sections of the coupled NP/Dye systems, along with the scattering and absorption cross sections of the uncoupled NP and uncoupled Dye under the same excitation source

-- The ground- and excited-state populations on the NP and Dye systems (coupled together) as a function of time during the simulation

-- The time-dependent dipole moment of the NP and Dye systems (coupled together)

