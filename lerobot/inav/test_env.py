import csv
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml
from environment.docking.Vdock import Vdock
from PIL import Image

from lerobot.inav.action_chunking_predict import Policy

from_episodes = False
all_episodes = False
render = True

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

# load best episode 
# if from_episodes:
#     # results_dir = '/media/jupiter/hddlinux/ascor/RESULTS/rl-visual-docking/landing/PPO_2025-02-07_10-38-02/PPO_2025-02-14_09-48-06/PPO_2025-02-16_11-31-16/PPO_environment.landing.landing.Landing_350db_00000_0_2025-02-16_11-31-16/checkpoint_000947/'
#     results_dir = '/media/jupiter/hddlinux/ascor/RESULTS/rl-visual-docking/AAS_big_sky_2023/'
#     # best_episode = load_dict(results_dir + 'best_episode.pkl')
#     # episodes = load_dict(results_dir + 'episodes_dict.pkl')
#     episodes = np.load(results_dir + 'episodes_dict.npy', allow_pickle=True).item()


ACTpolicy = Policy()

results_dir = '/home/andreascorsoglio/RESULTS/rl-visual-based-docking/AAS_big_sky_2023/'
episodes = np.load(results_dir + 'episodes_dict.npy', allow_pickle=True).item()

# import config file
with open('/home/andreascorsoglio/PROJECTS/PYTHON/rl-visual-based-docking/configs/config_galileogcrb_image.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)




episode0 = episodes[0]
state0 = {}
state0["x"] = np.array([episode0['x'][0]/1000, episode0['y'][0]/1000, episode0['z'][0]/1000, episode0['vx'][0]/1000, episode0['vy'][0]/1000, episode0['vz'][0]/1000])
state0["mass"] = episode0['mass'][0]
state0["q"] = np.array([episode0['q0'][0], episode0['q1'][0], episode0['q2'][0], episode0['q3'][0]])
state0["e"] = np.array([episode0['ex'][0]*np.pi/180, episode0['ey'][0]*np.pi/180, episode0['ez'][0]*np.pi/180])
state0["wb"] = np.array([episode0['wx'][0]*np.pi/180, episode0['wy'][0]*np.pi/180, episode0['wz'][0]*np.pi/180])


env_config = config['config']['env_config']
env = Vdock(env_config)
obs = env.reset(forced_state=state0)
next_state = env.state

# NOTE: at the start of the episode call:
# ACTpolicy.reset()

# visualize first image
if render and (env_config['obs_type'] == 2 or env_config['obs_type'] == 3):
    img = obs[0]
    # plt.imshow(img, cmap='gray')
    # plt.show()  

    fg = plt.figure()
    ax = fg.gca()
    h = ax.imshow(img)  # set initial display dimensions
        # plt.draw(), plt.pause(1e-4)



# load all images for first trajectory
images = []
for i in range(len(episode0['time'])):
    image = Image.open(f'/home/andreascorsoglio/RESULTS/rl-visual-based-docking/AAS_big_sky_2023/trajectories/imgs/img_traj_0_step_{i}.png')
    images.append(image)

img_loaded = images[0]

vector_obs = np.array([
                state0['x'][0]*1000, state0['x'][1]*1000, state0['x'][2]*1000,
                state0['x'][3]*1000, state0['x'][4]*1000, state0['x'][5]*1000,
                state0['q'][0], state0['q'][1], state0['q'][2], state0['q'][3],
                state0['wb'][0]*180/np.pi, state0['wb'][1]*180/np.pi, state0['wb'][2]*180/np.pi,
            ])

img = Image.fromarray(img)


################## TESTING WITH TRAINING IMAGES ##################

# img_loaded_array = np.array(img_loaded)
# img_rendered_array = np.array(img)

# # get difference between two images
# import cv2
# # diff = np.abs(np.array(img_loaded)[:,:,0] - np.array(img)[:,:,0]).astype(np.float16)
# diff = cv2.subtract(np.array(img_loaded)[:,:,0], np.array(img)[:,:,0])
# # diff_img = Image.fromarray(diff)

# ## plot images in two subplots
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.imshow(np.array(img_loaded)[:,:,0])
# ax1.set_title('Loaded Image')
# ax2.imshow(np.array(img)[:,:,0])
# ax2.set_title('Current Image')
# ax3.imshow(diff)
# ax3.set_title('Difference Image')
# plt.show()

# obs_loaded = {
#     "state": vector_obs,
#     "image": img_loaded
# }

# obs = {
#     "state": vector_obs,
#     "image": img
# }


# action_loaded = ACTpolicy.run_inference(obs_loaded)
# action_rendered = ACTpolicy.run_inference(obs)

###################################################################
    
times = []
states = []
rewards = []
actions = []
if from_episodes:
    if not all_episodes: # take best episode only
        episodes = [best_episode]
    for j, episode in enumerate(episodes.values()):
        with open(f'/media/jupiter/hddlinux/ascor/RESULTS/rl-visual-docking/AAS_big_sky_2023/trajectories/states/trajectory_{j}.csv', mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',')

            writer.writerow(['time', 'x', 'y', 'z', 'vx', 'vy', 'vz','mass','q0','q1','q2','q3','w1','w2','w3','Tx','Ty','Tz','Lx','Ly','Lz'])

            for i in range(len(episode['x'])):

                frameStart = time.time()

                prev_state = next_state 

                # control from loaded episode
                T = np.array([episode['Tx'][i], episode['Ty'][i], episode['Tz'][i]])
                L = np.array([episode['Lx'][i], episode['Ly'][i], episode['Lz'][i]])
                Tnorm = np.linalg.norm(T) + np.linalg.norm(L)
                control = [T,L,Tnorm]

                next_state = {}
                next_state["step"] = i
                next_state["t"] = episode['time'][i]
                next_state["x"] = np.array([episode['x'][i], episode['y'][i], episode['z'][i], episode['vx'][i], episode['vy'][i], episode['vz'][i]])
                next_state["mass"] = episode['mass'][i]
                next_state["q"] = np.array([episode['q0'][i], episode['q1'][i], episode['q2'][i], episode['q3'][i]])
                next_state["e"] = np.array([episode['ex'][i]*np.pi/180, episode['ey'][i]*np.pi/180, episode['ez'][i]*np.pi/180])
                next_state["wb"] = np.array([episode['wx'][i]*np.pi/180, episode['wy'][i]*np.pi/180, episode['wz'][i]*np.pi/180])
                obs = env.get_observation(next_state)
                reward, done = env.collect_reward(prev_state, next_state, control)
                rewards.append(reward)

                img = obs[0]

                # visualize image
                if render and (env_config['obs_type'] == 2 or env_config['obs_type'] == 3):
                    h.set_data(img)
                    plt.draw(), plt.pause(0.1)

                frameTime = time.time() - frameStart
                times.append(frameTime)
                states.append(next_state)


                # write csv file
                writer.writerow([episode['time'][i], episode['x'][i], episode['y'][i], episode['z'][i], episode['vx'][i], episode['vy'][i], episode['vz'][i], episode['mass'][i],
                                 episode['q0'][i], episode['q1'][i], episode['q2'][i], episode['q3'][i],
                                 episode['wx'][i], episode['wy'][i], episode['wz'][i],
                                 episode['Tx'][i], episode['Ty'][i], episode['Tz'][i],
                                 episode['Lx'][i], episode['Ly'][i], episode['Lz'][i]])
                

                # save image
                frame = Image.fromarray(img)
                frame.save(f'/media/jupiter/hddlinux/ascor/RESULTS/rl-visual-docking/AAS_big_sky_2023/trajectories/imgs/img_traj_{j}_step_{i}.png')

                if done:
                    obs = env.reset()
                    next_state = env.state
                    print(f'Done: step {i}')
                    break

        if j == 99:
            break
else:
    for i in range(len(episode0['time'])-1):
        frameStart = time.time()

        prev_state = next_state 

        # action = env.action_space.sample()

        # use state and image from environment
        vector_obs = np.array([
            prev_state['x'][0]*1000, prev_state['x'][1]*1000, prev_state['x'][2]*1000,
            prev_state['x'][3]*1000, prev_state['x'][4]*1000, prev_state['x'][5]*1000,
            prev_state['q'][0], prev_state['q'][1], prev_state['q'][2], prev_state['q'][3],
            prev_state['wb'][0]*180/np.pi, prev_state['wb'][1]*180/np.pi, prev_state['wb'][2]*180/np.pi,
        ])
        img_obs = obs[0]

        # use state and images from file
        # vector_obs = np.array([episode0['x'][i], episode0['y'][i], episode0['z'][i], episode0['vx'][i], episode0['vy'][i], episode0['vz'][i],
        #                         episode0['q0'][i], episode0['q1'][i], episode0['q2'][i], episode0['q3'][i],
        #                         episode0['wx'][i], episode0['wy'][i], episode0['wz'][i]])
        # img_obs = images[i]


        obs_dict = {
            "state": vector_obs,
            "image": img_obs
        }
        
        
        action_chunk = ACTpolicy.run_inference(obs_dict)

        # test with action from file
        # action = [episode0['Tx'][i], episode0['Ty'][i], episode0['Tz'][i], episode0['Lx'][i], episode0['Ly'][i], episode0['Lz'][i]]

        dt = episode0['time'][i+1] - episode0['time'][i]

        # Loop over action_chunk
        for j in range(len(action_chunk)):
            action = action_chunk[j]
            control = env.get_control(action, next_state)
            next_state = env.next_state(next_state, control, time_step=dt) # state is in km km/s and rad rad/s
            obs = env.get_observation(next_state)
            reward, done = env.collect_reward(prev_state, next_state, control)


        # actions.append(action)
        # control = env.get_control(action, next_state)
        # next_state = env.next_state(next_state, control, time_step=dt) # state is in km km/s and rad rad/s
        # obs = env.get_observation(next_state)
        # reward, done = env.collect_reward(prev_state, next_state, control)
        # rewards.append(reward)

        # visualize image
        if env_config['obs_type'] == 2 or env_config['obs_type'] == 3 and render:
            img = obs[0]
            h.set_data(img)
            plt.draw(), plt.pause(1e-4)

        frameTime = time.time() - frameStart
        times.append(frameTime)
        states.append(next_state)

        if done:
            obs = env.reset()
            next_state = env.state

            print(f'Done: step {i}')

            break

# print(times)
print(f'Average time: {np.mean(times)}')

print(np.sum(rewards))


# plot state along trajectory in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([states[i]['x'][0] for i in range(len(states))], [states[i]['x'][1] for i in range(len(states))], [states[i]['x'][2] for i in range(len(states))])
ax.scatter(0, 0, 0, c='r', marker='o')


# plot euler angles
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([states[i]['e'][0] for i in range(len(states))], label='phi')
ax.plot([states[i]['e'][1] for i in range(len(states))], label='theta')
ax.plot([states[i]['e'][2] for i in range(len(states))], label='psi')
plt.legend()

# plot quaternions 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([states[i]['q'][0] for i in range(len(states))], label='q0')
ax.plot([states[i]['q'][1] for i in range(len(states))], label='q1')
ax.plot([states[i]['q'][2] for i in range(len(states))], label='q2')
ax.plot([states[i]['q'][3] for i in range(len(states))], label='q3')
plt.legend()


# plot actions T and L
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot([actions[i][0] for i in range(len(actions))], label='Tx')
ax1.plot([actions[i][1] for i in range(len(actions))], label='Ty')
ax1.plot([actions[i][2] for i in range(len(actions))], label='Tz')
ax1.legend()
ax2.plot([actions[i][3] for i in range(len(actions))], label='Lx')
ax2.plot([actions[i][4] for i in range(len(actions))], label='Ly')
ax2.plot([actions[i][5] for i in range(len(actions))], label='Lz')
ax2.legend()
plt.show()