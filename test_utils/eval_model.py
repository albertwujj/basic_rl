def eval_model(model, env):
    totalRet = 0
    eps = 1000

    for i in range(eps):
        done = False
        obs = env.reset()
        while not done:
            a = model.choose_act([obs])[0][0]
            obs, rew, done, _ = env.step(a)
            totalRet += rew
    return totalRet / eps
