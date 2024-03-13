from typing import Optional

# def check_optimizer_args(self) -> dict:
#         if self.optimizer_args is None:
#             optimizer_args = {}
#         else:
#             optimizer_args = self.optimizer_args.copy()
#             for arg in (
#                 "iters",
#                 "uniset",
#                 "pop_size",
#             ):
#                 assert (
#                     arg not in optimizer_args
#                 ), f"Do not set '{arg}' in 'optimizer_args'. Instead, use the arguments of the class."
#             for arg in (
#                 "fitness_function",
#                 "fitness_function_args",
#                 "genotype_to_phenotype",
#                 "genotype_to_phenotype_args",
#                 "minimization",
#                 "init_population",
#                 "optimal_value",
#             ):
#                 assert (
#                     arg not in optimizer_args
#                 ), f"Do not set '{arg}' to 'optimizer_args'. It is defined automatically."

#         return optimizer_args


#     def check_optimizer_args(self) -> dict:

#         if self.optimizer_args is None:
#             optimizer_args = {"iters": 30, "pop_size": 100}
#         else:
#             optimizer_args = self.optimizer_args.copy()
#             for arg in (
#                 "fitness_function",
#                 "iters",
#                 "pop_size",
#                 "uniset",
#                 "genotype_to_phenotype",
#                 "minimization",
#             ):
#                 assert (
#                     arg not in optimizer_args.keys()
#                 ), f"""Do not set the "{arg}"
#                 to the "optimizer_args". It is defined automatically"""
#         return optimizer_args

#     def check_weights_optimizer_args(self) -> dict:

#         if self.weights_optimizer_args is None:
#             weights_optimizer_args = {"iters": 100, "pop_size": 100}
#         else:
#             weights_optimizer_args = self.weights_optimizer_args.copy()
#             for arg in (
#                 "fitness_function",
#                 "iters",
#                 "pop_size",
#                 "left",
#                 "right",
#                 "str_len",
#                 "genotype_to_phenotype",
#                 "minimization",
#             ):
#                 assert (
#                     arg not in weights_optimizer_args.keys()
#                 ), f"""Do not set the "{arg}"
#                 to the "weights_optimizer_args". It is defined automatically"""
#         return weights_optimizer_args


def check_optimizer_args(
    optimizer_args, args_auto_defined: Optional[list] = None, args_in_class: Optional[list] = None
) -> None:
    if args_auto_defined is not None:
        for arg in args_auto_defined:
            assert (
                arg not in optimizer_args.keys()
            ), f"""Do not set the "{arg}"
            to the "weights_optimizer_args". It is defined automatically"""

    if args_in_class is not None:
        for arg in args_in_class:
            assert (
                arg not in optimizer_args.keys()
            ), f"Do not set '{arg}' in 'optimizer_args'. Instead, use the arguments of the class."
