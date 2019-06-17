# Example of using Quantum Machine Learning (QML) for Molecular Dynamics
simulation using the ASE Package with visualization in Narupa VR

Using `Makefile` as a log file on setup.

Setting up training data and environment

    make
    source activate ./env
    make protocol-narupa

Training on the data

    make train

Testing that the model will produce forces

    make test

Will MD work?

    make run_md

And lastly, serve a molecule for Narupa

    make run_vr





