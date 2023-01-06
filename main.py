if __name__ == '__main__':
    from metrics import *
    #example()
    from pyg_models.pyg_runner import PygModelRunner
    pyg_runner = PygModelRunner(verbose=True)
    pyg_runner.run_all()

    # from stellargraph_models.model_runner import StellarGraphModelRunner
    # stellargraph_runner = StellarGraphModelRunner()
    # stellargraph_runner.run_all()
    #
    # from dgl_models.model_runner import DglModelRunner
    # dgl_runner = DglModelRunner()
    # dgl_runner.run_all()
