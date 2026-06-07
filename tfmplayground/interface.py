import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from tfmplayground.models.nanotabpfn import NanoTabPFNModel
from tfmplayground.utils import get_default_device
from pfns.bar_distribution import BarDistribution

def init_model_from_state_dict_file(file_path):
    """
    reads model architecture from state dict, instantiates the architecture and loads the weights
    """
    state_dict = torch.load(file_path, map_location=torch.device("cpu"))
    model = NanoTabPFNModel(
        num_attention_heads=state_dict["architecture"]["num_attention_heads"],
        embedding_size=state_dict["architecture"]["embedding_size"],
        mlp_hidden_size=state_dict["architecture"]["mlp_hidden_size"],
        num_layers=state_dict["architecture"]["num_layers"],
        num_outputs=state_dict["architecture"]["num_outputs"],
    )
    model.load_state_dict(state_dict["model"])
    return model


# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = (
            pd.to_numeric(X[col], errors="coerce").notna().sum()
        )  # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            ("to_pandas", FunctionTransformer(to_pandas)),  # to apply pd.to_numeric of pandas
            ("to_numeric", FunctionTransformer(to_numeric)),  # in case numeric columns are stored as strings
            (
                "imputer",
                SimpleImputer(strategy="mean", add_indicator=True),
            ),  # median might be better because of outliers
        ]
    )
    cat_transformer = Pipeline(
        [
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", num_transformer, num_mask), ("cat", cat_transformer, cat_mask)]
    )
    return preprocessor


class NanoTabPFNClassifier:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        device: None | str | torch.device = None,
        num_mem_chunks: int = 8,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "checkpoints/nanotabpfn.pth"
            if not os.path.isfile(model):
                os.makedirs("checkpoints", exist_ok=True)
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """stores X_train and y_train for later use, also computes the highest class number occuring in num_classes"""
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """calls predit_proba and picks the class with the highest probability for each datapoint"""
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model(
                (x, y), train_test_split_index=len(self.X_train), num_mem_chunks=self.num_mem_chunks
            ).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, : self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


class ExtendedFullSupportBarDistribution(BarDistribution):
    @staticmethod
    def halfnormal_with_p_weight_before(range_max, p=0.5):
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(
            torch.tensor(p)
        )
        return torch.distributions.HalfNormal(s)

    def forward(
        self, logits, y, mean_prediction_logits=None
    ):  # gives the negative log density (the _loss_), y: T x B, logits: T x B x self.num_bars
        assert self.num_bars > 1
        y = y.clone().view(len(y), -1)  # no trailing one dimension
        ignore_loss_mask = self.ignore_init(y)  # alters y
        target_sample = self.map_to_bucket_idx(y)  # shape: T x B (same as y)
        target_sample.clamp_(0, self.num_bars - 1)

        assert (
            logits.shape[-1] == self.num_bars
        ), f"{logits.shape[-1]} vs {self.num_bars}"
        assert (target_sample >= 0).all() and (
            target_sample < self.num_bars
        ).all(), f"y {y} not in support set for borders (min_y, max_y) {self.borders}"
        assert (
            logits.shape[-1] == self.num_bars
        ), f"{logits.shape[-1]} vs {self.num_bars}"
        # ignore all position with nan values

        scaled_bucket_log_probs = self.compute_scaled_log_probs(logits)

        assert len(scaled_bucket_log_probs) == len(target_sample), (
            len(scaled_bucket_log_probs),
            len(target_sample),
        )
        log_probs = scaled_bucket_log_probs.gather(
            -1, target_sample.unsqueeze(-1)
        ).squeeze(-1)

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )

        log_probs[target_sample == 0] += side_normals[0].log_prob(
            (self.borders[1] - y[target_sample == 0]).clamp(min=0.00000001)
        ) + torch.log(self.bucket_widths[0])
        log_probs[target_sample == self.num_bars - 1] += side_normals[1].log_prob(
            (y[target_sample == self.num_bars - 1] - self.borders[-2]).clamp(
                min=0.00000001
            )
        ) + torch.log(self.bucket_widths[-1])

        nll_loss = -log_probs

        if mean_prediction_logits is not None:
            assert (
                not ignore_loss_mask.any()
            ), "Ignoring examples is not implemented with mean pred."
            if not self.training:
                print("Calculating loss incl mean prediction loss for nonmyopic BO.")
            if not torch.is_grad_enabled():
                print(
                    "Warning: loss is not correct in absolute terms, only the gradient is right, when using `append_mean_pred`."
                )
            scaled_mean_log_probs = self.compute_scaled_log_probs(
                mean_prediction_logits
            )
            nll_loss = torch.cat(
                (nll_loss, self.mean_loss(logits, scaled_mean_log_probs)), 0
            )
            # ignore_loss_mask = torch.zeros_like(nll_loss, dtype=torch.bool)

        if self.smoothing:
            smooth_loss = -scaled_bucket_log_probs.mean(dim=-1)
            smoothing = self.smoothing if self.training else 0.0
            nll_loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss

        if ignore_loss_mask.any():
            nll_loss[ignore_loss_mask] = 0.0

        return nll_loss

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return p @ bucket_means.to(logits.device)

    def mean_of_square(self, logits):
        """
        Computes E[x^2].
        :param logits: Output of the model.
        """
        left_borders = self.borders[:-1]
        right_borders = self.borders[1:]
        bucket_mean_of_square = (
            left_borders.square()
            + right_borders.square()
            + left_borders * right_borders
        ) / 3.0
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_mean_of_square[0] = (
            side_normals[0].variance
            + (-side_normals[0].mean + self.borders[1]).square()
        )
        bucket_mean_of_square[-1] = (
            side_normals[1].variance
            + (self.borders[-2] + side_normals[1].mean).square()
        )
        p = torch.softmax(logits, -1)
        return p @ bucket_mean_of_square

    def pi(
        self, logits, best_f, maximize=True
    ):  # logits: evaluation_points x batch x feature_dim
        """
        Acquisition Function: Probability of Improvement
        :param logits: as returned by Transformer (evaluation_points x batch x feature_dim)
        :param best_f: best evaluation so far (the incumbent)
        :param maximize: whether to maximize
        :return: utility
        """
        assert maximize is True
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(
                logits[..., 0].shape, best_f, device=logits.device
            )  # evaluation_points x batch
        assert (
            best_f.shape == logits[..., 0].shape
        ), f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"
        p = torch.softmax(logits, -1)  # evaluation_points x batch
        border_widths = self.borders[1:] - self.borders[:-1]
        factor = 1.0 - ((best_f[..., None] - self.borders[:-1]) / border_widths).clamp(
            0.0, 1.0
        )  # evaluation_points x batch x num_bars

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        position_in_side_normals = (
            -(best_f - self.borders[1]).clamp(max=0.0),
            (best_f - self.borders[-2]).clamp(min=0.0),
        )  # evaluation_points x batch
        factor[..., 0] = 0.0
        factor[..., 0][position_in_side_normals[0] > 0.0] = side_normals[0].cdf(
            position_in_side_normals[0][position_in_side_normals[0] > 0.0]
        )
        factor[..., -1] = 1.0
        factor[..., -1][position_in_side_normals[1] > 0.0] = 1.0 - side_normals[1].cdf(
            position_in_side_normals[1][position_in_side_normals[1] > 0.0]
        )
        return (p * factor).sum(-1)

    def ei_for_halfnormal(self, scale, best_f, maximize=True):
        """
        This is the EI for a standard normal distribution with mean 0 and variance `scale` times 2.
        Which is the same as the half normal EI.
        I tested this with MC approximation:
        ei_for_halfnormal = lambda scale, best_f: (torch.distributions.HalfNormal(torch.tensor(scale)).sample((10_000_000,))- best_f ).clamp(min=0.).mean()
        print([(ei_for_halfnormal(scale,best_f), FullSupportBarDistribution().ei_for_halfnormal(scale,best_f)) for scale in [0.1,1.,10.] for best_f in [.1,10.,4.]])
        :param scale:
        :param best_f:
        :param maximize:
        :return:
        """
        assert maximize
        mean = torch.tensor(0.0)
        u = (mean - best_f) / scale
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        try:
            ucdf = normal.cdf(u)
        except ValueError:
            print(f"u: {u}, best_f: {best_f}, scale: {scale}")
            raise
        updf = torch.exp(normal.log_prob(u))
        normal_ei = scale * (updf + u * ucdf)
        return 2 * normal_ei

    def ei(
        self, logits, best_f, maximize=True
    ):  # logits: evaluation_points x batch x feature_dim
        if torch.isnan(logits).any():
            raise ValueError(f"logits contains NaNs: {logits}")
        bucket_diffs = self.borders[1:] - self.borders[:-1]
        assert maximize
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(logits[..., 0].shape, best_f, device=logits.device)
        assert (
            best_f.shape == logits[..., 0].shape
        ), f"best_f.shape: {best_f.shape}, logits.shape: {logits.shape}"

        best_f_per_logit = best_f[..., None].repeat(
            *[1] * len(best_f.shape), logits.shape[-1]
        )
        clamped_best_f = best_f_per_logit.clamp(self.borders[:-1], self.borders[1:])

        # true bucket contributions
        bucket_contributions = (
            (self.borders[1:] ** 2 - clamped_best_f**2) / 2
            - best_f_per_logit * (self.borders[1:] - clamped_best_f)
        ) / bucket_diffs

        # extra stuff for continuous
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        position_in_side_normals = (
            -(best_f - self.borders[1]).clamp(max=0.0),
            (best_f - self.borders[-2]).clamp(min=0.0),
        )  # evaluation_points x batch

        bucket_contributions[..., -1] = self.ei_for_halfnormal(
            side_normals[1].scale, position_in_side_normals[1]
        )

        bucket_contributions[..., 0] = self.ei_for_halfnormal(
            side_normals[0].scale, torch.zeros_like(position_in_side_normals[0])
        ) - self.ei_for_halfnormal(side_normals[0].scale, position_in_side_normals[0])

        p = torch.softmax(logits, -1)
        return torch.einsum("...b,...b->...", p, bucket_contributions)

    def _gaussian_interval_moments(self, mean, std, lower, upper):
        """
        Returns:
            Z  = ∫_lower^upper N(x; mean, std^2) dx
            M1 = ∫_lower^upper x N(x; mean, std^2) dx
            M2 = ∫_lower^upper x^2 N(x; mean, std^2) dx

        lower/upper may be +/- inf.
        """
        device = mean.device
        dtype = mean.dtype
        standard = torch.distributions.Normal(
            torch.zeros((), device=device, dtype=dtype),
            torch.ones((), device=device, dtype=dtype),
        )

        a = (lower - mean) / std
        b = (upper - mean) / std

        Phi_a = standard.cdf(a)
        Phi_b = standard.cdf(b)
        phi_a = torch.exp(standard.log_prob(a))
        phi_b = torch.exp(standard.log_prob(b))

        finite_a = torch.isfinite(a)
        finite_b = torch.isfinite(b)
        a_phi_a = torch.where(finite_a, a * phi_a, torch.zeros_like(a))
        b_phi_b = torch.where(finite_b, b * phi_b, torch.zeros_like(b))

        Z = Phi_b - Phi_a
        M1 = mean * Z + std * (phi_a - phi_b)
        M2 = (
            (mean.square() + std.square()) * Z
            + 2.0 * mean * std * (phi_a - phi_b)
            + std.square() * (a_phi_a - b_phi_b)
        )
        return Z, M1, M2

    def std(self, logits):
        """
        Exact standard deviation of the represented distribution.
        """
        mean = self.mean(logits)
        second_moment = self.mean_of_square(logits)
        var = (second_moment - mean.square()).clamp_min(0.0)
        return var.sqrt()

    def _gaussian_interval_moments(self, mean, std, lower, upper):
        """
        Computes

            Z  = ∫ N(x) dx
            M1 = ∫ x N(x) dx
            M2 = ∫ x² N(x) dx

        over [lower, upper].

        All arguments may be broadcastable tensors.
        """

        standard = torch.distributions.Normal(
            torch.zeros((), device=mean.device, dtype=mean.dtype),
            torch.ones((), device=mean.device, dtype=mean.dtype),
        )

        a = (lower - mean) / std
        b = (upper - mean) / std

        Phi_a = standard.cdf(a)
        Phi_b = standard.cdf(b)

        phi_a = torch.exp(standard.log_prob(a))
        phi_b = torch.exp(standard.log_prob(b))

        finite_a = torch.isfinite(a)
        finite_b = torch.isfinite(b)

        a_phi_a = torch.where(finite_a, a * phi_a, torch.zeros_like(a))
        b_phi_b = torch.where(finite_b, b * phi_b, torch.zeros_like(b))

        Z = Phi_b - Phi_a

        M1 = mean * Z + std * (phi_a - phi_b)

        M2 = (
            (mean.square() + std.square()) * Z
            + 2.0 * mean * std * (phi_a - phi_b)
            + std.square() * (a_phi_a - b_phi_b)
        )

        return Z, M1, M2

    def kl_div(self, logits, true_mean, true_std):
        """
        Exact reverse KL:

            KL( N(true_mean, true_std²) || q )

        where q is the distribution represented by logits.
        """

        device = logits.device
        dtype = logits.dtype

        true_mean = torch.as_tensor(true_mean, device=device, dtype=dtype)
        true_std = torch.as_tensor(true_std, device=device, dtype=dtype)

        if (true_std <= 0).any():
            raise ValueError("true_std must be positive.")

        log_p_logits = torch.log_softmax(logits, dim=-1)

        #
        # KL = -H(gaussian) - E_gaussian[log q]
        #

        entropy_gaussian = 0.5 * torch.log(2.0 * torch.pi * torch.e * true_std.square())

        expected_log_q = torch.zeros_like(true_mean)

        #
        # LEFT TAIL
        #
        cL = self.borders[1].to(device=device, dtype=dtype)

        left_scale = self.halfnormal_with_p_weight_before(
            self.bucket_widths[0].to(device=device, dtype=dtype)
        ).scale

        ZL, M1L, M2L = self._gaussian_interval_moments(
            true_mean,
            true_std,
            torch.full_like(true_mean, -torch.inf),
            torch.full_like(true_mean, cL),
        )

        moment2_left = cL.square() * ZL - 2.0 * cL * M1L + M2L

        left_const = log_p_logits[..., 0] + torch.distributions.HalfNormal(
            left_scale
        ).log_prob(torch.zeros_like(true_mean))

        expected_log_q += ZL * left_const - 0.5 * moment2_left / left_scale.square()

        #
        # INTERIOR BINS
        #
        if self.num_bars > 2:

            lowers = self.borders[1:-2].to(
                device=device,
                dtype=dtype,
            )

            uppers = self.borders[2:-1].to(
                device=device,
                dtype=dtype,
            )

            widths = uppers - lowers

            #
            # Broadcast:
            #
            # true_mean[:,None]
            # lowers[None,:]
            #
            # -> [N, K]
            #

            Zm, _, _ = self._gaussian_interval_moments(
                true_mean[..., None],
                true_std[..., None],
                lowers,
                uppers,
            )

            interior_log_density = log_p_logits[..., 1:-1] - widths.log()

            expected_log_q += (Zm * interior_log_density).sum(dim=-1)

        #
        # RIGHT TAIL
        #
        cR = self.borders[-2].to(
            device=device,
            dtype=dtype,
        )

        right_scale = self.halfnormal_with_p_weight_before(
            self.bucket_widths[-1].to(
                device=device,
                dtype=dtype,
            )
        ).scale

        ZR, M1R, M2R = self._gaussian_interval_moments(
            true_mean,
            true_std,
            torch.full_like(true_mean, cR),
            torch.full_like(true_mean, torch.inf),
        )

        moment2_right = M2R - 2.0 * cR * M1R + cR.square() * ZR

        right_const = log_p_logits[..., -1] + torch.distributions.HalfNormal(
            right_scale
        ).log_prob(torch.zeros_like(true_mean))

        expected_log_q += ZR * right_const - 0.5 * moment2_right / right_scale.square()

        return -entropy_gaussian - expected_log_q


class NanoTabPFNRegressor:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        dist: (
            FullSupportBarDistribution | ExtendedFullSupportBarDistribution | str | None
        ) = None,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = "checkpoints/nanotabpfn_regressor.pth"
            dist = "checkpoints/nanotabpfn_regressor_buckets.pth"
            if not os.path.isfile(model):
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print("No cached bucket edges found, downloading bucket edges.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth"
                )
                with open(dist, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(
                X, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            y_tensor = torch.tensor(
                y, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            logits = self.model(
                (X_tensor, y_tensor),
                train_test_split_index=len(self.X_train),
                num_mem_chunks=self.num_mem_chunks,
            ).squeeze(0)
            preds_n = self.dist.mean(logits)
            preds = preds_n * self.y_train_std + self.y_train_mean

        return preds.cpu().numpy()

    def predict_mean_std(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Similar to predict, but also computes the standard deviation of the output distributions.
        Returns both the means and standard deviations, renormalized to the original target scale.
        """
        if not isinstance(self.dist, ExtendedFullSupportBarDistribution):
            raise NotImplementedError(
                "predict_mean_std is only implemented for ExtendedFullSupportBarDistribution."
            )

        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(
                X, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            y_tensor = torch.tensor(
                y, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            logits = self.model(
                (X_tensor, y_tensor),
                train_test_split_index=len(self.X_train),
                num_mem_chunks=self.num_mem_chunks,
            ).squeeze(0)
            preds_n = self.dist.mean(logits)
            stds_n = self.dist.std(logits)

            preds = preds_n * self.y_train_std + self.y_train_mean
            stds = stds_n * self.y_train_std

        return preds.cpu().numpy(), stds.cpu().numpy()

    def predict_kl_divergence(
        self, X_test: np.ndarray, true_mean: np.ndarray, true_std: np.ndarray
    ) -> np.ndarray:
        """
        Computes the KL divergence between the predicted distributions and the true distributions for each test point.
        Assumes both the predicted and true distributions are Gaussian.
        """
        if not isinstance(self.dist, ExtendedFullSupportBarDistribution):
            raise NotImplementedError(
                "predict_kl_divergence is only implemented for ExtendedFullSupportBarDistribution."
            )

        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(
                X, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            y_tensor = torch.tensor(
                y, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            logits = self.model(
                (X_tensor, y_tensor),
                train_test_split_index=len(self.X_train),
                num_mem_chunks=self.num_mem_chunks,
            ).squeeze(0)

            kl_divs = self.dist.kl_div(logits, true_mean, true_std)
        return kl_divs.cpu().numpy()
