import numpy as np
import math
import random
import copy


# MARKET DATA (Portfolio Optimization Problem)
# Assets order: [Technology, Gold, Crypto, Bonds]


EXPECTED_RETURNS = np.array([0.15, 0.08, 0.30, 0.05])

COVARIANCE_MATRIX = np.array([
    [0.10,  0.02,  0.15,  0.01],
    [0.02,  0.05, -0.02,  0.01],
    [0.15, -0.02,  0.40,  0.02],
    [0.01,  0.01,  0.02,  0.02]
])


# OBJECTIVE FUNCTION (Negative Sharpe Ratio)


def sharpe_objective(weights):
    """
    Computes negative Sharpe ratio (for minimization).
    """
    weights = np.abs(weights)

    if np.sum(weights) == 0:
        return 1e6

    weights /= np.sum(weights)

    portfolio_return = np.dot(weights, EXPECTED_RETURNS)
    portfolio_volatility = np.sqrt(
        weights.T @ COVARIANCE_MATRIX @ weights
    )

    risk_free_rate = 0.02
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return -sharpe_ratio



# NEST REPRESENTATION


class Nest:
    """
    Represents a candidate solution (portfolio).
    """

    def __init__(self, dimension, bounds):
        self.dimension = dimension
        self.bounds = bounds
        self.solution = None
        self.fitness = None
        self.random_initialize()

    def random_initialize(self):
        """
        Initializes a random feasible portfolio.
        """
        lower, upper = self.bounds
        self.solution = np.random.uniform(lower, upper, self.dimension)
        self.solution /= np.sum(self.solution)
        self.evaluate()

    def evaluate(self):
        """
        Evaluates the current solution.
        """
        self.fitness = sharpe_objective(self.solution)

    def update(self, new_solution):
        """
        Updates the nest with a new solution after constraint handling.
        """
        new_solution = np.clip(new_solution, 0, 1)

        if np.sum(new_solution) > 0:
            new_solution /= np.sum(new_solution)

        self.solution = new_solution
        self.evaluate()



# TRUE CUCKOO SEARCH IMPLEMENTATION

class TrueCuckooSearch:
    """
    Canonical Cuckoo Search implementation:
    - Lévy flight (global exploration)
    - Local random walk
    - Adaptive nest abandonment
    - Elite preservation
    """

    def __init__(self, num_nests, dimension, bounds,
                 pa_max=0.25, pa_min=0.05):

        self.num_nests = num_nests
        self.dimension = dimension
        self.bounds = bounds

        self.pa_max = pa_max
        self.pa_min = pa_min
        self.beta = 1.5  # Lévy distribution parameter

        self.nests = [Nest(dimension, bounds) for _ in range(num_nests)]
        self.best_nest = min(self.nests, key=lambda n: n.fitness)

    # ---------------- Lévy Flight ----------------
    def levy_flight(self, solution):
        """
        Generates a Lévy flight step using Mantegna's algorithm.
        """
        sigma = (
            math.gamma(1 + self.beta) *
            math.sin(math.pi * self.beta / 2) /
            (math.gamma((1 + self.beta) / 2) *
             self.beta * 2 ** ((self.beta - 1) / 2))
        ) ** (1 / self.beta)

        u = np.random.normal(0, sigma, self.dimension)
        v = np.random.normal(0, 1, self.dimension)

        step = u / (np.abs(v) ** (1 / self.beta))
        return solution + 0.05 * step  # no attraction to global best

    # ---------------- Local Random Walk ----------------
    def local_random_walk(self, sol_a, sol_b):
        """
        Performs a local random walk between two solutions.
        """
        epsilon = np.random.rand(self.dimension)
        return sol_a + epsilon * (sol_b - sol_a)

    # ---------------- Optimization Loop ----------------
    def run(self, max_generations):

        for generation in range(max_generations):

            # Adaptive discovery probability
            pa = self.pa_max - (
                (self.pa_max - self.pa_min) * generation / max_generations
            )

            # ===== Global exploration (Lévy flights) =====
            for i in range(self.num_nests):
                new_solution = self.levy_flight(self.nests[i].solution)
                j = random.randint(0, self.num_nests - 1)

                if sharpe_objective(new_solution) < self.nests[j].fitness:
                    self.nests[j].update(new_solution)

            # ===== Local exploitation (random walks) =====
            for i in range(self.num_nests):
                j, k = random.sample(range(self.num_nests), 2)
                new_solution = self.local_random_walk(
                    self.nests[j].solution,
                    self.nests[k].solution
                )

                if sharpe_objective(new_solution) < self.nests[i].fitness:
                    self.nests[i].update(new_solution)

            # ===== Abandon worst nests =====
            self.nests.sort(key=lambda n: n.fitness)
            num_abandoned = int(self.num_nests * pa)

            for i in range(self.num_nests - num_abandoned, self.num_nests):
                self.nests[i].random_initialize()

            # ===== Elite preservation =====
            current_best = min(self.nests, key=lambda n: n.fitness)
            if current_best.fitness < self.best_nest.fitness:
                self.best_nest = copy.deepcopy(current_best)

        return self.best_nest


# RUN OPTIMIZATION

if __name__ == "__main__":
    # 1. Khởi tạo và thực thi
    optimizer = TrueCuckooSearch(num_nests=30, dimension=4, bounds=(0, 1))
    best_portfolio = optimizer.run(max_generations=150)

    # 2. Định nghĩa tên tài sản
    asset_names = ["Technology", "Gold", "Crypto", "Bonds"]

    # 3. Bắt đầu in kết quả chuyên nghiệp
    print("\n" + "="*45)
    print("   KẾT QUẢ TỐI ƯU HÓA DANH MỤC ĐẦU TƯ")
    print("      (Thuật toán True Cuckoo Search)")
    print("="*45)

    # In chi tiết phân bổ vốn
    print(f"{'Tài sản':<15} | {'Tỷ trọng (%)':>15}")
    print("-" * 45)
    for i, asset in enumerate(asset_names):
        weight_percentage = best_portfolio.solution[i] * 100
        print(f"{asset:<15} | {weight_percentage:>14.2f}%")

    # In các chỉ số đo lường hiệu quả
    print("-" * 45)
    
    # Tính toán lại các chỉ số để hiển thị
    final_return = np.dot(best_portfolio.solution, EXPECTED_RETURNS)
    final_risk = np.sqrt(best_portfolio.solution.T @ COVARIANCE_MATRIX @ best_portfolio.solution)
    
    print(f"Lợi nhuận kỳ vọng năm : {final_return * 100:.2f}%")
    print(f"Mức rủi ro (Volatility): {final_risk * 100:.2f}%")
    print(f"Chỉ số Sharpe tối ưu  : {-best_portfolio.fitness:.4f}")
    print("="*45)
    print("Ghi chú: Tỷ trọng đã được chuẩn hóa (Tổng = 100%)")
