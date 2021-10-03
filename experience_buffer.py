from collections import deque
import random


class ExperienceBuffer:

    def __init__(self, max_buffer_size, batch_size):
        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = deque([], maxlen=max_buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, grip_info\')'
        assert type(experience[5]) == bool
        self.experiences.push(experience)

    def get_batch(self):
        return [*zip(*(random.sample(self.experiences, self.batch_size)))]
