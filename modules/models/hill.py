from genessa.models.hill import HillCell
from .mutation import Mutation


class HillModel(HillCell, Mutation):
    """
    Class defines a cell with a single protein coding gene subject to negative feedback. Transcription and feedback terms are based on Hill kinetics.

    Attributes:

        name (str) - name of controlled gene

    Inherited Attributes:

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

    def __init__(self, name='X', k1=1, k_m=1, n=1, k2=1, g1=1, g2=1):
        """
        Instantiate the Hill model.

        Args:

            name (str) - name of controlled gene

            k1 (float) - maximal transcription rate

            k_m (float) - michaelis constant

            n (float) - hill coefficient

            k2 (float) - translation rate constant

            g1 (float) - transcript decay rate constant

            g2 (float) - protein decay rate constant

        """

        self.name = name

        # instantiate Hill cell with a single gene
        gene_kw = dict(k=k2, g1=g1, g2=g2)
        super().__init__(genes=(self.name,), I=1, **gene_kw)

        # add transcriptional activation by input
        self.add_transcription(self.name, ('IN',), k=k1, k_m=k_m, n=n)

    def add_transcriptional_feedback(self, k_m=1, n=1, **kwargs):
        """
        Adds transcriptional auto-repression.

        Args:

            k_m (float) - michaelis constant

            n (float) - hill coefficient

            kwargs: keyword arguments for repressor

        """
        self.add_transcriptional_repressor(
             actuators=(self.name,),
             target=self.name,
             k_m=k_m,
             n=n,
             **kwargs)

    def add_post_transcriptional_feedback(self,
                                     k=None,
                                     atp_sensitive=2,
                                     ribosome_sensitive=True,
                                     **kwargs):
        """
        Adds linear negative feedback applied to transcript level.

        Args:

            k (float) - rate parameter (feedback strength)

            atp_sensitive (int) - order of metabolism dependence

            ribosome_sensitive (bool) - scale rate parameter with ribosomes

            kwargs: keyword arguments for reaction

        """

        self.add_linear_feedback(
             sensor=self.name,
             target=self.name,
             mode='transcript',
             k=k,
             atp_sensitive=atp_sensitive,
             ribosome_sensitive=ribosome_sensitive,
             **kwargs)

    def add_post_translational_feedback(self,
                                     k=None,
                                     atp_sensitive=2,
                                     ribosome_sensitive=True,
                                     **kwargs):
        """
        Adds linear negative feedback applied to protein level.

        Args:

            k (float) - rate parameter (feedback strength)

            atp_sensitive (int) - order of metabolism dependence

            ribosome_sensitive (bool) - scale rate parameter with ribosomes

            kwargs: keyword arguments for reaction

        """

        self.add_linear_feedback(
             sensor=self.name,
             target=self.name,
             mode='protein',
             k=k,
             atp_sensitive=atp_sensitive,
             ribosome_sensitive=ribosome_sensitive,
             **kwargs)

    # def add_post_transcriptional_feedback(self,
    #                                  k=1,
    #                                  Kd=1,
    #                                  atp_sensitive=2,
    #                                  ribosome_sensitive=True,
    #                                  **kwargs):
    #     """
    #     Adds negative feedback applied to transcript level. Kinetics are based on a michaelis menten mechanism.

    #     Args:

    #         k (float) - maximum degradation rate (feedback strength)

    #         Kd (float) - michaelis constant

    #         atp_sensitive (int) - order of metabolism dependence

    #         ribosome_sensitive (bool) - scale rate parameter with ribosomes

    #         kwargs: keyword arguments for transcript degradation reaction

    #     """
    #     self.add_transcript_degradation(
    #         actuator=self.name,
    #         target=self.name,
    #         k=k,
    #         Kd=Kd,
    #         atp_sensitive=atp_sensitive,
    #         ribosome_sensitive=ribosome_sensitive,
    #         **kwargs)

    # def add_post_translational_feedback(self,
    #                                  k=1,
    #                                  Kd=1,
    #                                  atp_sensitive=2,
    #                                  ribosome_sensitive=True,
    #                                  **kwargs):
    #     """
    #     Adds negative feedback applied to protein level. Kinetics are based on a michaelis menten mechanism.

    #     Args:

    #         k (float) - maximum degradation rate (feedback strength)

    #         Kd (float) - michaelis constant

    #         atp_sensitive (int) - order of metabolism dependence

    #         ribosome_sensitive (bool) - scale rate parameter with ribosomes

    #         kwargs: keyword arguments for protein degradation reaction

    #     """

    #     self.add_protein_degradation(
    #         actuator=self.name,
    #         target=self.name,
    #         k=k,
    #         Kd=Kd,
    #         atp_sensitive=atp_sensitive,
    #         ribosome_sensitive=ribosome_sensitive,
    #         **kwargs)
