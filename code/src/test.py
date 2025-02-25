import gym

def main():
    env = gym.make('MiniHack-MyCustomEnv-v0')
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()

if __name__ == "__main__":
    main()
