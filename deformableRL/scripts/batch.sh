#!/bin/bash

# Define ranges for hyperparameters
actor_lr_values=(0.0001 0.0003 0.001)
critic_lr_values=(0.0001 0.00018 0.0003)
batch_size_values=(1024 2048 4096 8192 16384)
gradient_steps_values=(1 2 4 8)

# Loop through all combinations of hyperparameters
for actor_lr in "${actor_lr_values[@]}"
do
    for critic_lr in "${critic_lr_values[@]}"
    do
        for batch_size in "${batch_size_values[@]}"
        do
            for gradient_steps in "${gradient_steps_values[@]}"
            do
                for policy_delay in "${gradient_steps_values[@]}"
                do
                    # Print the current configuration
                    echo "Running training with:
                          actor_lr=$actor_lr,
                          critic_lr=$critic_lr,
                          batch_size=$batch_size,
                          gradient_steps=$gradient_steps,
                          policy_delay=$gradient_steps"
                    
                    # Run the training script
                    ~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh rlgames_train.py  headless=True \
                      train.params.config.actor_lr=$actor_lr \
                      train.params.config.critic_lr=$critic_lr \
                      train.params.config.batch_size=$batch_size \
                      train.params.config.gradient_steps=$gradient_steps \
                      train.params.config.policy_delay=$gradient_steps
                    
                    # Wait for the current training to finish before starting the next one
                    wait
                done
            done
        done
    done
done
