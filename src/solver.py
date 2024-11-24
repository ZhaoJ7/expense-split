import pandas as pd
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
import typing as tp


Person = tp.NewType("Person", str)
Amount = tp.NewType("Amount", float)
Shares = tp.NewType("Share", float)


@dataclass
class Transfer:
    from_: Person
    to: Person
    amount: Amount


@dataclass
class SolverResult:
    transfers: tp.List[Transfer]
    total_num_transfers: int
    total_transfer_amount: Amount


class Solver:

    def __init__(self, data: pd.DataFrame):
        # TODO: validate input data

        # ----- Creates some helper variables ----- #
        all_people = data.columns[2:].tolist()
        num_people = len(all_people)
        total_amount_paid = data["Amount"].sum().item()

        total_paid_by_person: dict[Person, Amount] = {}
        total_required_by_person: dict[Person, Amount] = {}

        for _, row in data.iterrows():
            person_paid: Person = row["Who paid"]
            amount_paid: Amount = row["Amount"]
            total_shares: Shares = row[all_people].sum()

            total_paid_by_person[person_paid] = (
                total_paid_by_person.get(person_paid, 0.0) + amount_paid
            )

            for person in all_people:
                person_shares: Shares = row[person]
                total_required_by_person[person] = (
                    total_required_by_person.get(person, 0.0)
                    + amount_paid * person_shares / total_shares
                )

        # ----- Create the optimisation problem ----- #
        # Matrix of boolean variables which indicates whether a transfer required between each party
        #  The row is the "from" party and the col is the "to" party
        #  Index is in the same order as all_names
        is_transfer = cp.Variable((num_people, num_people), boolean=True)

        # Float variable which indicates the transfer amount between parties
        transfer_amount = cp.Variable((num_people, num_people))

        # Objective is to 1. minimise the number of transfers required (higher priority)
        #  and 2. minimise the amount being transferred
        total_num_transfers = cp.sum(is_transfer)
        total_transfer_amount = cp.sum(transfer_amount)
        weight_for_amount = 1 / total_amount_paid
        obj = cp.Minimize(
            total_num_transfers + total_transfer_amount * weight_for_amount
        )

        # Create the constraints
        constraints = []

        # Transfer amounts must be positive
        constraints.append(transfer_amount >= 0)

        for i in range(num_people):
            for j in range(num_people):

                # Create constraint that you can't transfer to yourself
                if i == j:
                    constraints.append(is_transfer[i, j] == 0)
                    constraints.append(transfer_amount[i, j] == 0)

                # Creates a constraint that if no transfer is required, then the transfer amount must be 0
                constraints.append(
                    transfer_amount[i, j] <= is_transfer[i, j] * total_amount_paid
                )

                # Creates a constraint that if a transfer is required, then the transfer amount must be positive
                eps = 1e-2
                constraints.append(transfer_amount[i, j] >= is_transfer[i, j] * eps)

        # Creates constraints that: already paid + will transfer - will receive = required to pay
        for i, name in enumerate(all_people):
            total_out = cp.sum(transfer_amount[i, :])
            total_in = cp.sum(transfer_amount[:, i])
            constraints.append(
                total_paid_by_person.get(name, 0.0) + total_out - total_in
                == total_required_by_person.get(name, 0.0)
            )

        # Now create the problem
        prob = cp.Problem(objective=obj, constraints=constraints)

        # ----- Attach all created objects as attributes ----- #
        # 1. Helper variables
        self.all_people = all_people
        self.num_people = num_people
        self.total_amount_paid = total_amount_paid
        self.total_paid_by_person = total_paid_by_person
        self.total_required_by_person = total_required_by_person

        # 2. High-level solver attributes
        self.prob = prob
        self.obj = obj
        self.constraints = constraints
        self.solver_result: tp.Optional[SolverResult] = None

        # 3. Low level solver attributes
        self.total_num_transfers = total_num_transfers
        self.total_transfer_amount = total_transfer_amount
        self.is_transfer = is_transfer
        self.transfer_amount = transfer_amount

    def run(self, verbose: bool = True) -> SolverResult:
        self.prob.solve(verbose=verbose)

        is_transfer_matrix = self.is_transfer.value
        transfer_amount_matrix = self.transfer_amount.value

        transfers = []
        for i, from_person in enumerate(self.all_people):
            for j, to_person in enumerate(self.all_people):
                if not np.isclose(is_transfer_matrix[i, j], 0):
                    transfers.append(
                        Transfer(
                            from_=from_person,
                            to=to_person,
                            amount=transfer_amount_matrix[i, j].item(),
                        )
                    )

        solver_result = SolverResult(
            transfers=transfers,
            total_transfer_amount=self.total_transfer_amount.value.item(),
            total_num_transfers=self.total_num_transfers.value.item(),
        )
        self.solver_result = solver_result

        return solver_result


if __name__ == "__main__":
    from config import DATA_DIR

    data = pd.read_excel(DATA_DIR / "example.xlsx")
    solver_ = Solver(data)
    solver_res = solver_.run()
    print(solver_res)
