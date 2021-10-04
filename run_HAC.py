"""
"run_HAC.py" executes the training schedule for the agent.  By default, the agent will alternate between
exploration and testing phases.  The number of episodes in the exploration phase can be configured in section 3 of
"design_agent_and_env.py" file.  If the user prefers to only explore or only test, the user can enter the
command-line options ""--train_only" or "--test", respectively.  The full list of command-line options is available
in the "options.py" file.
"""
import logging
import os
import utils
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

NUM_BATCH = 1000
TEST_FREQ = 2

num_test_episodes = 100

logger = logging.getLogger(f'HAC.run_HAC')


def run_HAC(FLAGS, env, agent):
    # Print task summary
    utils.print_summary(FLAGS, env)
    logger.info(FLAGS)

    # Determine training mode. If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not FLAGS.test and not FLAGS.train_only:
        mix_train_test = True

    writer = SummaryWriter(log_dir=os.path.join('experiments', FLAGS.env))
    # If not retraining, restore weights
    # if we are not retraining from scratch, just restore weights
    if FLAGS.restore_file is not None:
        start_batch = utils.load_checkpoint(agent, FLAGS.model_dir, FLAGS.restore_file) + 1
    else:
        start_batch = 0

    for batch in trange(start_batch, NUM_BATCH):
        
        agent.penalize_subgoal_count = [0 for _ in range(FLAGS.layers)]
        agent.total_subgoal_test = [0 for _ in range(FLAGS.layers)]
        agent.total_transitions = [0 for _ in range(FLAGS.layers)]
        num_episodes = agent.other_params["num_exploration_episodes"]

        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and (batch + 1) % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            agent.FLAGS.test = True
            num_episodes = num_test_episodes

            # Reset successful episode counter
            successful_episodes = 0

        for episode in trange(num_episodes):

            # print(f'\nBatch {batch}, Episode {episode}')

            # Train for an episode
            success = agent.train(env, episode)

            if success:
                # print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))

                # Increment successful episode counter if applicable
                if mix_train_test and (batch + 1) % TEST_FREQ == 0:
                    successful_episodes += 1

        if not agent.FLAGS.test:
            for i in range(FLAGS.layers):
                logger.info(f'Level {i} penalize rate: {agent.penalize_subgoal_count[i] / agent.total_subgoal_test[i]}')
                logger.info(f'Level {i} total transitions: {agent.total_transitions[i]}')

        # Finish evaluating policy if tested prior batch
        if mix_train_test and (batch + 1) % TEST_FREQ == 0:
            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            logger.info(f'Batch {batch} success rate {success_rate: .3f}%')
            writer.add_scalar(f"{FLAGS.model}/success_rate", success_rate, batch)
            writer.flush()
            utils.save_checkpoint(agent, batch, success_rate, FLAGS.model_dir)
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")
