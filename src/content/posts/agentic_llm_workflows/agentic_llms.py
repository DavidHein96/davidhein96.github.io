import re
import sqlite3
from enum import Enum, StrEnum
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


class IHCResultPrimary(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    OTHER = "Other"


class IHCResultModifier(str, Enum):
    DIFFUSE = "Diffuse"
    BOX_LIKE = "Box like"
    CUP_LIKE = "Cup like"


class IHCTestName(str, Enum):
    BAP1 = "BAP1"
    CA_IX = "CA-IX"
    OTHER = "Other"


class IHCTest(BaseModel):
    specimen: str = Field(..., description="Specimen used for test")

    test_name: IHCTestName = Field(..., description="Name of test")
    test_name_other: Optional[str] = Field(
        None, description="Name of test if not in options"
    )

    test_result: IHCResultPrimary = Field(..., description="Test Result")
    test_result_modifier: Optional[IHCResultModifier] = Field(
        None, description="Test result modifier if applicable"
    )
    test_result_other: Optional[str] = Field(
        None, description="Result if not in options"
    )


class IHCReport(BaseModel):
    reasoning: str = Field(..., description="Summary of reasoning")
    test_and_results: Optional[list[IHCTest]] = Field(
        None, description="List of tests and results"
    )


# Here we use the handy OpenAI parse which lets us pass the pydantic
#   model directly, for vllm we would dump it first

client = OpenAI()
prompt = "Here is my pathology report..."

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format=IHCReport,
)

if response.choices[0].message.parsed is not None:
    parsed: IHCReport = response.choices[0].message.parsed


class IHCResultPrimary(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    OTHER = "Other"


class IHCResultModifier(str, Enum):
    DIFFUSE = "Diffuse"
    BOX_LIKE = "Box like"
    CUP_LIKE = "Cup like"


class IHCTestName(str, Enum):
    BAP1 = "BAP1"
    CA_IX = "CA-IX"
    OTHER = "Other"


def _normalize_key(s: str) -> str:
    """
    Lowercases, strips whitespace, hyphens,
    underscores, slashes, dots and paranthesis"""
    return re.sub(pattern=r"[\s\-\_./()]+", repl="", string=s).lower()


def _build_enum_lookup(enum_cls: type[StrEnum]) -> dict[str, str]:
    """Builds {normalized_key: enum_valie map}"""
    return {_normalize_key(m.value): m.value for m in enum_cls}


def normalize_enum_value(
    raw_value: object, enum_cls: type[StrEnum], lookup: dict[str, str]
) -> str:
    if isinstance(raw_value, enum_cls):
        return raw_value.value
    clean_name = str(raw_value).strip()
    match = lookup.get(_normalize_key(clean_name))
    return match if match is not None else clean_name


class IHCTest(BaseModel):
    specimen: str = Field(..., description="Specimen used for test")

    test_name: IHCTestName = Field(..., description="Name of test")
    test_name_other: Optional[str] = Field(
        None, description="Name of test if not in options"
    )

    test_result: IHCResultPrimary = Field(..., description="Test Result")
    test_result_modifier: Optional[IHCResultModifier] = Field(
        None, description="Test result modifier if applicable"
    )
    test_result_other: Optional[str] = Field(
        None, description="Result if not in options"
    )


# --- tool implementation ---
def record_ihc_test(**kwargs) -> tuple[str, IHCTest]:
    ihc_test = IHCTest(**kwargs)
    # Tool responses should be JSON-serializable strings for the model
    return json.dumps({"ok": True}), ihc_test


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_ihc_test",
            "description": "Record an IHC test result for a specimen",
            "parameters": IHCTest.model_json_schema(),  # pydantic v2
        },
    },
]


# Now a tool calling loop, ill just do pseudo code since its kinda long...
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
tests, failures = [], 0

for _ in range(6):
    msg = (
        client.chat.completions
        .create(model="gpt-5", messages=messages, tools=TOOLS)
        .choices[0]
        .message
    )
    messages.append(msg)

    if msg.tool_calls:
        for call in msg.tool_calls:
            try:
                out, t = record_ihc_test(**json.loads(call.function.arguments))
                tests.append(t)
            except ValidationError as e:
                out = json.dumps({"ok": False, "error": str(e)})
                failures += 1
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": out,
            })
        if failures >= 3:
            raise RuntimeError("too many validation failures")
        continue

    report = IHCReport.model_validate_json(msg.content)
    save_final_result(report, tests)
    break


def init_state_db() -> sqlite3.Connection:
    """
    In-memory state for ONE patient's extraction session.
    (You could swap ':memory:' for a real file if desired.)
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS therapy_lines (
            patient_id   TEXT NOT NULL,
            therapy_name TEXT NOT NULL,
            start_date   TEXT,   -- ISO: YYYY-MM-DD
            stop_date    TEXT,   -- ISO: YYYY-MM-DD
            stop_reason  TEXT,
            PRIMARY KEY (patient_id, therapy_name)
        )
        """
    )
    conn.commit()
    return conn


def get_active_therapies(conn: sqlite3.Connection, patient_id: str) -> List[str]:
    rows = conn.execute(
        """
        SELECT therapy_name
        FROM therapy_lines
        WHERE patient_id = ?
          AND start_date IS NOT NULL
          AND stop_date IS NULL
        """,
        (patient_id,),
    ).fetchall()
    return [r["therapy_name"] for r in rows]


def record_therapy_start(
    conn: sqlite3.Connection,
    patient_id: str,
    therapy_name: str,
    start_date: str,
) -> None:
    """
    Insert a line of therapy, or fill in the start_date if the row already exists.
    """
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO therapy_lines (patient_id, therapy_name, start_date)
                VALUES (?, ?, ?)
                """,
                (patient_id, therapy_name, start_date),
            )
    except sqlite3.IntegrityError:
        # Row already exists. Only set start_date if it wasn't set yet.
        with conn:
            conn.execute(
                """
                UPDATE therapy_lines
                SET start_date = ?
                WHERE patient_id = ?
                  AND therapy_name = ?
                  AND start_date IS NULL
                """,
                (start_date, patient_id, therapy_name),
            )


def record_therapy_stop(
    conn: sqlite3.Connection,
    patient_id: str,
    therapy_name: str,
    stop_date: str,
    stop_reason: str,
) -> None:
    """
    Enforces a simple temporal constraint:
    you can only stop a therapy that is currently active.
    """
    row = conn.execute(
        """
        SELECT start_date
        FROM therapy_lines
        WHERE patient_id = ?
          AND therapy_name = ?
          AND start_date IS NOT NULL
          AND stop_date IS NULL
        """,
        (patient_id, therapy_name),
    ).fetchone()

    if row is None:
        active = get_active_therapies(conn, patient_id)
        # Raise a "tool constraint error" you feed back to the model
        # the same way you'd feed back a Pydantic ValidationError.
        raise ValueError(
            "Cannot record stop: no active therapy start found. "
            f"patient_id={patient_id!r}, therapy_name={therapy_name!r}. "
            f"Active therapies: {active}"
        )

    with conn:
        conn.execute(
            """
            UPDATE therapy_lines
            SET stop_date = ?, stop_reason = ?
            WHERE patient_id = ?
              AND therapy_name = ?
            """,
            (stop_date, stop_reason, patient_id, therapy_name),
        )


def execute_tool_call(tool_name: str, tool_args: dict) -> str:
    try:
        if tool_name == "record_therapy_start":
            record_therapy_start(**tool_args)
            return "OK"
        elif tool_name == "record_therapy_stop":
            record_therapy_stop(**tool_args)
            return "OK"
        # ... other tools
    except (ValueError, ValidationError) as e:
        # Return error string — caller feeds this back to the model
        return f"CONSTRAINT_ERROR: {str(e)}"
