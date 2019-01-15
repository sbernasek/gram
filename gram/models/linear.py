from genessa.models.linear import LinearCell
from .mutation import Mutation


class LinearModel(LinearCell, Mutation):
    """
    Class defines a cell with a single protein coding gene subject to negative feedback. All reaction rates are based on linear propensity functions.

    Attributes:

        name (str) - name of controlled gene

    Inherited Attributes:

        genes (dict) - {name: node_id} pairs - unused by default

        transcripts (dict) - {name: node_id} pairs

        proteins (dict) - {name: node_id} pairs

        phosphorylated (dict) - {name: node_id} pairs

        nodes (np.ndarray) - vector of node indices

        node_key (dict) - {state dimension: node id} pairs

        reactions (list) - list of reaction objects

        stoichiometry (np.ndarray) - stoichiometric coefficients, (N,M)

        N (int) - number of nodes

        M (int) - number of reactions

        I (int) - number of inputs

    """

    def __init__(self, name='X', k0=1, k1=1, k2=1, g0=1, g1=1, g2=1):
        """
        Instantiate the linear model.

        Args:

            name (str) - name of controlled gene

            k0 (float) - gene activation rate constant

            k1 (float) - transcription rate constant

            k2 (float) - translation rate constant

            g0 (float) - gene decay rate constant

            g1 (float) - transcript decay rate constant

            g2 (float) - protein decay rate constant

        """

        self.name = name

        # instantiate linear cell with a single gene activated by the input
        gene_kw = dict(k1=k1, k2=k2, g0=g0, g1=g1, g2=g2)
        super().__init__(genes=(self.name,), I=1, **gene_kw)

        # add transcriptional activation by input
        self.add_activation(gene=self.name, activator='IN', k=k0)

    def add_transcriptional_feedback(self,
                                     k=None,
                                     atp_sensitive=2,
                                     carbon_sensitive=2,
                                     ribosome_sensitive=1,
                                     **kwargs):
        """
        Adds linear negative feedback applied to activated-DNA level.

        Args:

            k (float) - rate parameter (feedback strength)

            atp_sensitive (int) - order of metabolism dependence

            carbon_sensitive (int) - order of carbon availability dependence

            ribosome_sensitive (int) - order of ribosome dependence

            kwargs: keyword arguments for reaction

        """

        self.add_linear_feedback(
             sensor=self.name,
             target=self.name,
             mode='gene',
             k=k,
             atp_sensitive=atp_sensitive,
             carbon_sensitive=carbon_sensitive,
             ribosome_sensitive=ribosome_sensitive,
             **kwargs)

    def add_post_transcriptional_feedback(self,
                                     k=None,
                                     atp_sensitive=2,
                                     carbon_sensitive=2,
                                     ribosome_sensitive=1,
                                     **kwargs):
        """
        Adds linear negative feedback applied to transcript level.

        Args:

            k (float) - rate parameter (feedback strength)

            atp_sensitive (int) - order of metabolism dependence

            carbon_sensitive (int) - order of carbon availability dependence

            ribosome_sensitive (int) - order of ribosome dependence

            kwargs: keyword arguments for reaction

        """

        self.add_linear_feedback(
             sensor=self.name,
             target=self.name,
             mode='transcript',
             k=k,
             atp_sensitive=atp_sensitive,
             carbon_sensitive=carbon_sensitive,
             ribosome_sensitive=ribosome_sensitive,
             **kwargs)

    def add_post_translational_feedback(self,
                                     k=None,
                                     atp_sensitive=2,
                                     carbon_sensitive=2,
                                     ribosome_sensitive=1,
                                     **kwargs):
        """
        Adds linear negative feedback applied to protein level.

        Args:

            k (float) - rate parameter (feedback strength)

            atp_sensitive (int) - order of metabolism dependence

            carbon_sensitive (int) - order of carbon availability dependence

            ribosome_sensitive (int) - order of ribosome dependence

            kwargs: keyword arguments for reaction

        """

        self.add_linear_feedback(
             sensor=self.name,
             target=self.name,
             mode='protein',
             k=k,
             atp_sensitive=atp_sensitive,
             carbon_sensitive=carbon_sensitive,
             ribosome_sensitive=ribosome_sensitive,
             **kwargs)
