import argparse
import os
import numpy as np
import pandas as pd


class ArgsModel:

    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_rd", type=int, default=100,
            help="the number of rounds.")
        parser.add_argument("--tableNum", type=float, default=2.1,
            help="the table to replicate. valid value: 2.1, 3.1, 3.2, 3.3.")

        parser.add_argument("--C0", type=float, default=0.0,
            help="initial State of the category C0 (in proportion).")
        parser.add_argument("--C2", type=float, default=0.0,
            help="initial State of the category C2 (in proportion).")
        parser.add_argument("--C3", type=float, default=0.0,
            help="initial State of the category C3 (in proportion).")
        parser.add_argument("--D0", type=float, default=0.0,
            help="initial State of the category D0 (in proportion).")
        parser.add_argument("--D1", type=float, default=0.0,
            help="initial State of the category D1 (in proportion).")
        parser.add_argument("--D3", type=float, default=0.0,
            help="initial State of the category D3 (in proportion).")
        
        self.parser = parser
    

    @staticmethod
    def set_table_init_val(args, tableNum):
        if tableNum not in {2.1, 3.1, 3.2, 3.3}:
            raise ValueError("Invalid tableNum.")
        
        args.tableNum = tableNum
        if tableNum == 2.1 or tableNum == 3.1:
            args.C0 = 0.25
            args.C2 = 0.25
            args.C3 = 0.0
            args.D0 = 0.0
            args.D1 = 0.25
            args.D3 = 0.25
            if tableNum == 2.1:
                args.n_rd = 100
            elif tableNum == 3.1:
                args.n_rd = 8955
        
        elif tableNum == 3.2:
            args.C0 = 0.09
            args.C2 = 0.01
            args.C3 = 0.0
            args.D0 = 0.0
            args.D1 = 0.81
            args.D3 = 0.09
            args.n_rd = 9022
        
        elif tableNum == 3.3:
            args.C0 = 0.01
            args.C2 = 0.0
            args.C3 = 0.0
            args.D0 = 0.0
            args.D1 = 0.0
            args.D3 = 0.99
            args.n_rd = 99579
        
        return args


    def get_args(self, tableNum=None):
        args = self.parser.parse_args()
        if tableNum is None:
            return self.set_table_init_val(args, args.tableNum)
        else:
            return self.set_table_init_val(args, tableNum)


C0, C2, C3 = 0, 1, 2
D0, D1, D3 = 3, 4, 5

class Simulation(object):

    @staticmethod
    def update_state(s: np.ndarray):
        assert s.size == 6
        new_s = np.zeros(6)
        
        # C0
        new_s[C0] = s[C0]*s[D0]
        new_s[C0] += s[C2]*(s[D0]+s[D1])
        new_s[C0] += s[C3]*(s[D0]+s[D1]+s[D3])

        # C2
        new_s[C2] = (s[C0]+s[C2]+s[C3])**2

        # C3
        new_s[C3] = (s[D0]+s[D1])*(s[C2]+s[C3])

        # D0
        new_s[D0] = s[C0]*(s[D1]+s[D3])
        new_s[D0] += s[C2]*s[D3]

        # D1
        new_s[D1] = (s[D0]+s[D1]+s[D3])**2

        # D3
        new_s[D3] = (s[D0]+s[D1])*s[C0]
        new_s[D3] += s[D3]*(s[C0]+s[C2]+s[C3])

        new_s[np.argmax(new_s)] += 1-np.sum(new_s) # adjust to make its sum be exactly 1.0

        return new_s


    @staticmethod
    def state_to_results(res, state, first=False):
        p_c = state[C0] + state[C2] + state[C3]
        mean_c = (0*state[C0] + 2*state[C2] + 3*state[C3]) / p_c
        mean_d = (0*state[D0] + 1*state[D1] + 3*state[D3]) / (1 - p_c)
        res_state = np.array([list(state)+[p_c, mean_c, mean_d]])
        res_state = np.transpose(res_state)
        if first:
            return res_state
        else:
            return np.concatenate((res, res_state), axis=1)
        

    @staticmethod
    def print_final_result(result: np.ndarray):
        print("++++++++++++++++++++++++")
        print("=========RESULT=========")
        print("C0: ", result[C0][-1])
        print("C2: ", result[C3][-1])
        print("C3: ", result[C2][-1])
        print("D0: ", result[D0][-1])
        print("D1: ", result[D1][-1])
        print("D3: ", result[D3][-1])
        print("C:  ", result[6][-1])
        print("C mean fitness: ", result[7][-1])
        print("D mean fitness: ", result[8][-1])
        print("++++++++++++++++++++++++")


    @staticmethod
    def run_simulation(args: argparse.ArgumentParser, save_as_csv=True,
        output_dir=os.path.join(os.getcwd(), "csv_results"), log_f=10) -> np.ndarray:
        """
        Return a ndarray of shape (9, args.n_rd).
        The meaning of the rows is as following:
            - the proportion of C0
            - the proportion of C2
            - the proportion of C3
            - the proportion of D0
            - the proportion of D1
            - the proportion of D3
            - the proportion of C
            - mean fitness of cooperators
            - mean fitness of defectors
        """

        print(args)

        log_rd = [int(args.n_rd*i/log_f) for i in range(1, log_f+1)]
        log_rd_ctr = 0

        state = np.array([args.C0, args.C2, args.C3, args.D0, args.D1, args.D3])
        result = Simulation.state_to_results(None, state, first=True)

        for rd_ctr in range(1, args.n_rd+1):
            if rd_ctr == log_rd[log_rd_ctr]:
                print("round {}% | {}/{}".format(int(100*rd_ctr/args.n_rd), rd_ctr, args.n_rd))
                log_rd_ctr += 1
            state = Simulation.update_state(state)
            result = Simulation.state_to_results(result, state)

        header = ["Initial State"] + ["Round {}".format(rd_ctr) for rd_ctr in range(1, args.n_rd+1)]
        fn1 = "tableNum_{}_nROUND_{}".format(args.tableNum, args.n_rd)
        fn2 = "C0_{:.2f}_C2_{:.2f}_C3_{:.2f}_D0_{:.2f}_D1_{:.2f}_D3_{:.2f}".format( \
            args.C0, args.C2, args.C3, args.D0, args.D1, args.D3)
        fpath = os.path.join(output_dir, "{}_{}.csv".format(fn1, fn2))
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pd.DataFrame(result).to_csv(fpath, header=header)

        print("The csv file successfully saved at {}".format(fpath))
        Simulation.print_final_result(result)

        return result


if __name__ == "__main__":
    args_hdl = ArgsModel()
    args = args_hdl.get_args()

    result = Simulation.run_simulation(args)