# A Micromechanical Model for Fracture Process Zone Generation and Microcrack Interactions with a Main Fracture 

This repositry contains the simulation code for my 2020 Thesis on the topic of Microcracks in Quasi-Brittle Material
Author: Dominic Walker

## Abstract ##

This paper develops a micromechanical model for the development and evolution of microcrack populations ahead of a main fracture. The model combines analytical solutions for crack stress fields, establishes a growth law for individual microcracks, and accounts for main fracture–microcrack interactions through the superposition of stresses. 

The model was first used to analyse the steady-state statistics of microcrack geometry and density ahead of the main fracture. Then, interactions of the main fracture with individual microcracks were considered and compared to interactions with populations of microcracks. Finally, a sensitivity analysis was conducted to assess the influence of the microcrack population on the roughness of the main fracture path. 

It is demonstrated that the distributions of lengths and orientations for microcracks passing points ahead of the main fracture are well-defined and show strong potential for analytical description. The study of interactions highlighted the importance of considering the microcrack population as a whole, not simply the effects of microcracks nearest to the main fracture tip. The significance of the population increases as microcrack density increases. Additionally, the roughness of the fracture path can be related to the size and density of the microcrack population. Larger microcrack populations increase fracture path roughness significantly, while increasing microcrack density increases roughness at a decreasing rate.


## Simulation Code ##
There are two classes of simulation code in this repository.
1. Microcrack populations are generated in the main fracture's stress field and microcrack populatoin statistics are studied (Refer: "microcrack-generation").
2. An interaction mechanism is developed to consider the two way interaction between a main fracture and the population of microcracks (Refer: "microcrack-interation-with-main-fracture").

