import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from collections import deque
import random

import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPG():
    def __init__(self, n_features, a_low, a_high, learning_rate_actor, learning_rate_critic, n_actor_hidden, n_critic_hidden, gamma=0.9, noise_varience=3, soft_replace=0.1, memory_size=1000, batch_size=128):
        self.n_features = n_features
        self.a_low = a_low
        self.a_high = a_high
        self.lr_a = learning_rate_actor
        self.lr_c = learning_rate_critic
        self.n_actor_hidden = n_actor_hidden
        self.n_critic_hidden = n_critic_hidden
        self.gamma = gamma
        self.noise_var = noise_varience
        self.soft_replace = soft_replace
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = batch_size

        self.noise = OrnsteinUhlenbeckNoise(size=1) 
        
        self.actor_model = self.build_actor()
        self.target_actor_model = self.build_actor()
        self.critic_model = self.build_critic()
        self.target_critic_model = self.build_critic()
        
        self.actor_optimizer = optimizers.Adam(lr=self.lr_a)
        self.critic_optimizer = optimizers.Adam(lr=self.lr_c)
        
        self.update_network_parameters(tau=1)
        
    def build_actor(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.n_features,)))
        model.add(layers.Dense(self.n_actor_hidden, activation='relu'))
        model.add(layers.Dense(1, activation='tanh'))
        model.add(layers.Lambda(lambda x: x * (self.a_high - self.a_low) / 2 + (self.a_high + self.a_low) / 2))
        return model
    
    def build_critic(self):
        state_input = layers.Input(shape=(self.n_features,))
        action_input = layers.Input(shape=(1,))
        concat = layers.Concatenate()([state_input, action_input])
        
        hidden = layers.Dense(self.n_critic_hidden, activation='relu')(concat)
        output = layers.Dense(1, activation=None)(hidden)
        model = models.Model([state_input, action_input], output)
        return model
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.soft_replace
            
        weights = []
        targets = self.target_actor_model.weights
        for i, weight in enumerate(self.actor_model.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor_model.set_weights(weights)

        weights = []
        targets = self.target_critic_model.weights
        for i, weight in enumerate(self.critic_model.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_model.set_weights(weights)
        
    def store_transition(self, state, action, reward, next_state):
        transition = (state, action, reward, next_state)
        self.memory.append(transition)

    def choose_action(self, state):
        state = np.reshape(state, [1, self.n_features])
        action = self.actor_model.predict(state)[0]
        return np.clip(action, self.a_low, self.a_high)
    
    def choose_action_with_OrnsteinUhlenbeckNoise(self, state):
        state = np.reshape(state, [1, self.n_features])
        action = self.actor_model.predict(state)[0]
        noise = self.noise.noise()  # Generate noise
        action = np.clip(action + noise, self.a_low, self.a_high)  # Add noise to the action
        return action

        
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states)
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor_model(next_states, training=True)
            y = rewards + self.gamma * self.target_critic_model([next_states, target_actions], training=True)
            critic_value = self.critic_model([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
        
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, training=True)
            critic_value = self.critic_model([states, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
            
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
        
        self.update_network_parameters()
