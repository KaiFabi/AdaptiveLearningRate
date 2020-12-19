import numpy as np

class GradientDescent(object):

    def __init__(self, f, x_init, y_init, n_iterations, eta, alpha, optimizer):

        self.gamma = 0.5
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.epsilon = 1e-8

        self.f = f
        self.x = x_init
        self.y = y_init
        self.n_iterations = n_iterations
        self.alpha = alpha

        self.eta_x = eta
        self.eta_y = eta

        self.dx_old = 0.0
        self.dy_old = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_x_old = 0.0
        self.v_y_old = 0.0
        self.m_x_old = 0.0
        self.m_y_old = 0.0

        self.i = 0

        self.x_history = list()
        self.y_history = list()
        self.eta_x_history = list()
        self.eta_y_history = list()

        if optimizer == "GD" or optimizer == "GD+":
            self.optimizer = self.gradient_descent 
        elif optimizer == "GDM" or optimizer == "GDM+":
            self.optimizer = self.gradient_descent_momentum
        elif optimizer == "NAG" or optimizer == "NAG+":
            self.optimizer = self.nesterov_accelerated_gradient
        elif optimizer == "Adam" or optimizer == "Adam+":
            self.optimizer = self.gradient_descent_adam
        else:
            raise Exception("Error: No such optimizer")

    def minimize(self):

        for self.i in range(self.n_iterations):
            self.x_history.append(self.x)
            self.y_history.append(self.y)
            self.eta_x_history.append(self.eta_x)
            self.eta_y_history.append(self.eta_y)
            self.optimizer()
        return self.x_history, self.y_history, self.eta_x_history, self.eta_y_history

    def gradient_descent(self):

        dx = self.dfdx(self.x, self.y)
        dy = self.dfdy(self.x, self.y)

        self.eta_x += self.alpha * dx * self.dx_old
        self.eta_y += self.alpha * dy * self.dy_old

        self.x -= self.eta_x * dx
        self.y -= self.eta_y * dy

        self.dx_old = dx
        self.dy_old = dy

    def gradient_descent_momentum(self):

        dx = self.dfdx(self.x, self.y)
        dy = self.dfdy(self.x, self.y)

        self.eta_x += self.alpha * dx * self.dx_old
        self.eta_y += self.alpha * dy * self.dy_old

        v_x = self.gamma * self.v_x_old + self.eta_x * dx
        v_y = self.gamma * self.v_y_old + self.eta_y * dy

        self.x -= v_x
        self.y -= v_y

        self.v_x_old = v_x
        self.v_y_old = v_y

        self.dx_old = dx
        self.dy_old = dy

    def nesterov_accelerated_gradient(self):

        dx = self.dfdx(self.x + self.gamma * self.v_x_old, self.y + self.gamma * self.v_y_old)
        dy = self.dfdy(self.x + self.gamma * self.v_x_old, self.y + self.gamma * self.v_y_old)

        self.eta_x += self.alpha * dx * self.dx_old
        self.eta_y += self.alpha * dy * self.dy_old

        v_x = self.gamma * self.v_x_old + self.eta_x * dx
        v_y = self.gamma * self.v_y_old + self.eta_y * dy

        self.x -= v_x
        self.y -= v_y

        self.v_x_old = v_x
        self.v_y_old = v_y

        self.dx_old = dx
        self.dy_old = dy

    def gradient_descent_adam(self):

        dx = self.dfdx(self.x, self.y)
        dy = self.dfdy(self.x, self.y)

        m_x = self.beta_1 * self.m_x_old + (1.0 - self.beta_1) * dx
        m_y = self.beta_1 * self.m_y_old + (1.0 - self.beta_1) * dy

        v_x = self.beta_2 * self.v_x_old + (1.0 - self.beta_2) * dx * dx
        v_y = self.beta_2 * self.v_y_old + (1.0 - self.beta_2) * dy * dy

        m_x_hat = m_x / (1.0 - self.beta_1**(self.i+1))
        m_y_hat = m_y / (1.0 - self.beta_1**(self.i+1))

        v_x_hat = v_x / (1.0 - self.beta_2**(self.i+1))
        v_y_hat = v_y / (1.0 - self.beta_2**(self.i+1))

        self.eta_x += self.alpha * dx * self.dx_old
        self.eta_y += self.alpha * dy * self.dy_old

        self.x -= (self.eta_x / (np.sqrt(v_x_hat) + self.epsilon)) * m_x_hat
        self.y -= (self.eta_y / (np.sqrt(v_y_hat) + self.epsilon)) * m_y_hat

        self.m_x_old = m_x
        self.m_y_old = m_y

        self.v_x_old = v_x
        self.v_y_old = v_y

        self.dx_old = dx
        self.dy_old = dy

    def dfdx(self, x, y, h=1e-9):
        return 0.5 * (self.f(x+h, y) - self.f(x-h, y)) / h

    def dfdy(self, x, y, h=1e-9):
        return 0.5 * (self.f(x, y+h) - self.f(x, y-h)) / h

