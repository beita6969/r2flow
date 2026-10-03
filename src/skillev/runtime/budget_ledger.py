from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from .contracts import BudgetReservation, BudgetSettlement, BudgetVector


class ReservationState(StrEnum):
    RESERVED = "reserved"
    SETTLED = "settled"


class BudgetExceededError(RuntimeError):
    pass


class DuplicateBudgetReservationError(RuntimeError):
    pass


class DuplicateBudgetSettlementError(RuntimeError):
    pass


class UnknownBudgetReservationError(RuntimeError):
    pass


class BudgetSettlementError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    reservation: BudgetReservation
    state: ReservationState
    settlement: BudgetSettlement | None

    @classmethod
    def reserved(cls, reservation: BudgetReservation) -> LedgerEntry:
        return cls(
            reservation=reservation,
            state=ReservationState.RESERVED,
            settlement=None,
        )

    def settled(self, settlement: BudgetSettlement) -> LedgerEntry:
        if self.state is not ReservationState.RESERVED:
            raise DuplicateBudgetSettlementError(
                f"reservation {self.reservation.reservation_id!r} is already settled"
            )
        if settlement.reservation_id != self.reservation.reservation_id:
            raise BudgetSettlementError("settlement targets another reservation")
        if not settlement.actual.fits_within(self.reservation.maximum):
            raise BudgetSettlementError("exact measured usage exceeds the admitted maximum")
        return LedgerEntry(
            reservation=self.reservation,
            state=ReservationState.SETTLED,
            settlement=settlement,
        )


class BudgetLedger:
    def __init__(
        self,
        *,
        run_id: str,
        attempt_id: str,
        cap: BudgetVector,
    ) -> None:
        if type(run_id) is not str or not run_id.strip():
            raise ValueError("run_id must be non-empty text")
        if type(attempt_id) is not str or not attempt_id.strip():
            raise ValueError("attempt_id must be non-empty text")
        if not isinstance(cap, BudgetVector):
            raise TypeError("cap must be BudgetVector")
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.cap = cap
        self._entries: dict[str, LedgerEntry] = {}
        self._reserved = BudgetVector()
        self._settled = BudgetVector()
        self._lock = Lock()

    @property
    def reserved(self) -> BudgetVector:
        with self._lock:
            return self._reserved

    @property
    def settled(self) -> BudgetVector:
        with self._lock:
            return self._settled

    @property
    def available(self) -> BudgetVector:
        with self._lock:
            return self.cap.subtract(self._reserved.add(self._settled))

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries[key] for key in sorted(self._entries))

    def reserve(self, reservation: BudgetReservation) -> LedgerEntry:
        if not isinstance(reservation, BudgetReservation):
            raise TypeError("reservation must be BudgetReservation")
        if reservation.run_id != self.run_id:
            raise ValueError("reservation belongs to another run")
        if reservation.attempt_id != self.attempt_id:
            raise ValueError("reservation belongs to another attempt")
        with self._lock:
            reservation_id = reservation.reservation_id
            if reservation_id in self._entries:
                raise DuplicateBudgetReservationError(
                    f"reservation ID {reservation_id!r} was reused"
                )
            proposed = self._reserved.add(self._settled).add(reservation.maximum)
            if not proposed.fits_within(self.cap):
                raise BudgetExceededError(
                    f"reservation {reservation_id!r} exceeds the attempt budget"
                )
            entry = LedgerEntry.reserved(reservation)
            self._entries[reservation_id] = entry
            self._reserved = self._reserved.add(reservation.maximum)
            return entry

    def settle(self, settlement: BudgetSettlement) -> LedgerEntry:
        if not isinstance(settlement, BudgetSettlement):
            raise TypeError("settlement must be BudgetSettlement")
        with self._lock:
            entry = self._entries.get(settlement.reservation_id)
            if entry is None:
                raise UnknownBudgetReservationError(
                    f"unknown reservation {settlement.reservation_id!r}"
                )
            if entry.state is ReservationState.SETTLED:
                raise DuplicateBudgetSettlementError(
                    f"reservation {settlement.reservation_id!r} was settled twice"
                )
            updated = entry.settled(settlement)
            self._reserved = self._reserved.subtract(entry.reservation.maximum)
            self._settled = self._settled.add(settlement.actual)
            self._entries[settlement.reservation_id] = updated
            return updated

    def assert_fully_settled(self) -> None:
        with self._lock:
            live = tuple(
                entry.reservation.reservation_id
                for entry in self._entries.values()
                if entry.state is ReservationState.RESERVED
            )
        if live:
            raise RuntimeError(f"attempt has unsettled reservations: {live!r}")

    def restore_completed(self, entries: Iterable[LedgerEntry]) -> None:
        entries = tuple(entries)
        with self._lock:
            imported: dict[str, LedgerEntry] = {}
            actual = BudgetVector()
            for entry in entries:
                reservation = entry.reservation
                if reservation.run_id != self.run_id or reservation.attempt_id != self.attempt_id:
                    raise ValueError("saved call belongs to a different run or attempt")
                if entry.state is not ReservationState.SETTLED or entry.settlement is None:
                    raise ValueError("only completed call charges can be restored")
                LedgerEntry.reserved(reservation).settled(entry.settlement)
                identity = reservation.reservation_id
                if identity in self._entries or identity in imported:
                    raise DuplicateBudgetReservationError("saved call was already consumed")
                imported[identity] = entry
                actual = actual.add(entry.settlement.actual)
            total = self._settled.add(actual)
            if not total.add(self._reserved).fits_within(self.cap):
                raise BudgetExceededError("saved call charges exceed the attempt budget")
            self._entries.update(imported)
            self._settled = total


__all__ = [
    "BudgetExceededError",
    "BudgetLedger",
    "BudgetSettlementError",
    "DuplicateBudgetReservationError",
    "DuplicateBudgetSettlementError",
    "LedgerEntry",
    "ReservationState",
    "UnknownBudgetReservationError",
]
