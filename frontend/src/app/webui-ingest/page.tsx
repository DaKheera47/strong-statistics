"use client";

import { useEffect, useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import Link from "next/link";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ThemeToggle } from "@/components/theme-toggle";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Loader2,
  UploadCloud,
} from "lucide-react";

interface CsvPreview {
  headers: string[];
  rows: string[][];
  totalRows: number;
}

interface DialogState {
  status: "success" | "error";
  title: string;
  description: string;
  details?: string;
}

const PREVIEW_ROW_LIMIT = 25;
const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(
  /\/$/,
  ""
);

function parseCsv(text: string, limit: number): CsvPreview {
  const rows: string[][] = [];
  let cell = "";
  let row: string[] = [];
  let insideQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];

    if (char === '"') {
      const nextChar = text[i + 1];
      if (insideQuotes && nextChar === '"') {
        cell += '"';
        i += 1;
      } else {
        insideQuotes = !insideQuotes;
      }
    } else if (char === "," && !insideQuotes) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !insideQuotes) {
      if (char === "\r" && text[i + 1] === "\n") {
        i += 1;
      }
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }

  if (rows.length > 0) {
    const lastRow = rows[rows.length - 1];
    if (lastRow.every((value) => value === "")) {
      rows.pop();
    }
  }

  if (rows.length === 0) {
    return { headers: [], rows: [], totalRows: 0 };
  }

  const [rawHeader, ...dataRows] = rows;
  const headers = rawHeader.map((header, index) =>
    index === 0 ? header.replace(/^\ufeff/, "").trim() : header.trim()
  );
  const previewRows = dataRows.slice(0, limit);

  return {
    headers,
    rows: previewRows,
    totalRows: dataRows.length,
  };
}

function formatBytes(size: number): string {
  if (Number.isNaN(size) || size <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(
    Math.floor(Math.log(size) / Math.log(1024)),
    units.length - 1
  );
  const value = size / 1024 ** exponent;
  return `${value.toFixed(value < 10 && exponent > 0 ? 1 : 0)} ${
    units[exponent]
  }`;
}

export default function IngestPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [token, setToken] = useState<string>("");
  const [parseError, setParseError] = useState<string | null>(null);
  const [isParsing, setIsParsing] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [dialogState, setDialogState] = useState<DialogState | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [ingestEndpoint, setIngestEndpoint] = useState<string>(
    API_BASE ? `${API_BASE}/ingest` : "/ingest"
  );

  useEffect(() => {
    const storedToken = window.localStorage.getItem("strong-statistics-token");
    if (storedToken) {
      setToken(storedToken);
    }
  }, []);

  useEffect(() => {
    if (!API_BASE) {
      setIngestEndpoint(`${window.location.origin}/ingest`);
    }
  }, []);

  useEffect(() => {
    if (token) {
      window.localStorage.setItem("strong-statistics-token", token);
    } else {
      window.localStorage.removeItem("strong-statistics-token");
    }
  }, [token]);

  const fileMeta = useMemo(() => {
    if (!selectedFile) return null;
    return {
      name: selectedFile.name,
      size: formatBytes(selectedFile.size),
      lastModified: new Date(selectedFile.lastModified).toLocaleString(),
      type: selectedFile.type || "text/csv",
    };
  }, [selectedFile]);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setSelectedFile(file ?? null);
    setPreview(null);
    setParseError(null);

    if (!file) {
      event.target.value = "";
      return;
    }

    setIsParsing(true);
    try {
      const text = await file.text();
      const parsed = parseCsv(text, PREVIEW_ROW_LIMIT);
      if (!parsed.headers.length) {
        setParseError("Could not detect any header rows in this CSV file.");
      } else if (!parsed.totalRows) {
        setParseError("No data rows found. Is this the correct Strong export?");
      }
      setPreview(parsed);
    } catch (error) {
      setParseError(
        error instanceof Error ? error.message : "Failed to read file"
      );
    } finally {
      setIsParsing(false);
    }
    event.target.value = "";
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setParseError(null);
    setFileInputKey((key) => key + 1);
  };

  const handleIngest = async () => {
    if (!selectedFile || !token) {
      setDialogState({
        status: "error",
        title: "Missing details",
        description:
          "Provide both a CSV export and the ingest token before uploading.",
      });
      setDialogOpen(true);
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setIsIngesting(true);
    try {
      const response = await fetch(
        `${ingestEndpoint}?token=${encodeURIComponent(token)}`,
        {
          method: "POST",
          body: formData,
        }
      );

      const contentType = response.headers.get("content-type") ?? "";
      const isJson = contentType.includes("application/json");

      if (!response.ok) {
        let detail = response.statusText;
        if (isJson) {
          const errorBody = (await response.json().catch(() => null)) as
            | { detail?: string }
            | null;
          if (errorBody?.detail) {
            detail = errorBody.detail;
          }
        } else {
          const text = await response.text().catch(() => "");
          if (text) {
            detail = text.slice(0, 200);
          }
        }
        throw new Error(detail || "Upload failed");
      }

      if (isJson) {
        const result = (await response.json().catch(() => null)) as
          | { stored?: string; rows?: number }
          | null;
        if (
          typeof result?.rows === "number" &&
          typeof result?.stored === "string"
        ) {
          setDialogState({
            status: "success",
            title: "Upload complete",
            description: `${result.rows} rows ingested successfully.`,
            details: `Stored as ${result.stored}`,
          });
        } else {
          setDialogState({
            status: "success",
            title: "Upload complete",
            description: "CSV ingested successfully.",
          });
        }
      } else {
        const text = await response.text().catch(() => "");
        setDialogState({
          status: "success",
          title: "Upload complete",
          description: text || "CSV ingested successfully.",
        });
      }
      setPreview(null);
      setSelectedFile(null);
      setFileInputKey((key) => key + 1);
      setParseError(null);
    } catch (error) {
      setDialogState({
        status: "error",
        title: "Ingestion failed",
        description:
          error instanceof Error
            ? error.message
            : "Unexpected error during ingestion.",
      });
    } finally {
      setDialogOpen(true);
      setIsIngesting(false);
    }
  };

  return (
    <div className='flex min-h-screen flex-col bg-background text-foreground [scrollbar-gutter:stable]'>
      <div className='flex-1 py-8'>
        <header className='mb-8 w-full'>
          <div className='flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between'>
            <div>
              <h1 className='text-3xl font-bold mb-2'>CSV Ingestion</h1>
              <p className='text-muted-foreground max-w-xl'>
                Preview your Strong export before sending it to the ingestion
                API. We keep uploads local, just like the dashboard.
              </p>
            </div>
            <div className='flex items-center gap-3'>
              <Link href='/'>
                <Button variant='outline'>
                  <ArrowLeft className='size-4' /> Dashboard
                </Button>
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className='grid w-full gap-6'>
          <Card>
            <CardHeader>
              <CardTitle className='flex items-center gap-2 text-xl'>
                <UploadCloud className='size-5 text-primary' />
                Upload Strong CSV
              </CardTitle>
              <CardDescription>
                Select the export file and supply the ingest token configured in
                your `.env`.
              </CardDescription>
            </CardHeader>
            <CardContent className='space-y-6'>
              <div className='grid gap-4 md:grid-cols-2'>
                <div className='space-y-2'>
                  <Label htmlFor='csv-file'>CSV export</Label>
                  <Input
                    key={fileInputKey}
                    id='csv-file'
                    type='file'
                    accept='.csv,text/csv'
                    onChange={handleFileChange}
                  />
                  {fileMeta && (
                    <div className='text-xs text-muted-foreground space-y-1'>
                      <p>
                        <span className='font-medium text-foreground'>
                          File:
                        </span>{" "}
                        {fileMeta.name}
                      </p>
                      <p>
                        <span className='font-medium text-foreground'>
                          Size:
                        </span>{" "}
                        {fileMeta.size}
                      </p>
                      <p>
                        <span className='font-medium text-foreground'>
                          Modified:
                        </span>{" "}
                        {fileMeta.lastModified}
                      </p>
                    </div>
                  )}
                </div>
                <div className='space-y-2'>
                  <Label htmlFor='ingest-token'>Ingest token</Label>
                  <Input
                    id='ingest-token'
                    type='password'
                    placeholder='INGEST_TOKEN value'
                    value={token}
                    onChange={(event) => setToken(event.target.value.trim())}
                  />
                  <p className='text-xs text-muted-foreground'>
                    Stored locally so you don&rsquo;t have to paste it every
                    visit.
                  </p>
                </div>
              </div>

              {isParsing && (
                <Alert variant='default'>
                  <Loader2 className='size-4 animate-spin' />
                  <AlertTitle>Parsing preview</AlertTitle>
                  <AlertDescription>
                    Reading the first {PREVIEW_ROW_LIMIT} rows from the CSV
                    export.
                  </AlertDescription>
                </Alert>
              )}

              {parseError && (
                <Alert variant='destructive'>
                  <AlertCircle className='size-4' />
                  <AlertTitle>Preview error</AlertTitle>
                  <AlertDescription>{parseError}</AlertDescription>
                </Alert>
              )}

              {!parseError && preview && !isParsing && (
                <Alert variant='default'>
                  <CheckCircle2 className='size-4' />
                  <AlertTitle>Preview ready</AlertTitle>
                  <AlertDescription>
                    Showing the first {preview.rows.length} of{" "}
                    {preview.totalRows} sets detected from the export.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
            <CardFooter className='flex items-center justify-end gap-3'>
              <div className='flex items-center gap-2'>
                <Button
                  variant='ghost'
                  disabled={!selectedFile && !preview}
                  onClick={handleReset}
                >
                  Clear
                </Button>
                <Button
                  onClick={handleIngest}
                  disabled={isIngesting || !selectedFile || isParsing}
                >
                  {isIngesting && <Loader2 className='size-4 animate-spin' />}
                  Ingest CSV
                </Button>
              </div>
            </CardFooter>
          </Card>

          {preview && !parseError && !isParsing && (
            <Card>
              <CardHeader>
                <CardTitle>Preview</CardTitle>
                <CardDescription>
                  Inspect columns and the first few sets before committing them
                  to the database.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Accordion
                  type='multiple'
                  defaultValue={["overview", "rows"]}
                >
                  <AccordionItem value='overview'>
                    <AccordionTrigger>File overview</AccordionTrigger>
                    <AccordionContent className='space-y-2'>
                      {fileMeta && (
                        <div className='grid gap-2 sm:grid-cols-2 text-sm'>
                          <p>
                            <span className='text-muted-foreground'>Name:</span>{" "}
                            {fileMeta.name}
                          </p>
                          <p>
                            <span className='text-muted-foreground'>
                              Rows detected:
                            </span>{" "}
                            {preview.totalRows}
                          </p>
                          <p>
                            <span className='text-muted-foreground'>Size:</span>{" "}
                            {fileMeta.size}
                          </p>
                          <p>
                            <span className='text-muted-foreground'>Type:</span>{" "}
                            {fileMeta.type}
                          </p>
                        </div>
                      )}
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value='columns'>
                    <AccordionTrigger>Column headers</AccordionTrigger>
                    <AccordionContent>
                      <div className='flex flex-wrap gap-2 text-xs'>
                        {preview.headers.map((header, index) => (
                          <span
                            key={`${header}-${index}`}
                            className='rounded-full bg-muted px-3 py-1 text-foreground'
                          >
                            {header || "(empty)"}
                          </span>
                        ))}
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value='rows'>
                    <AccordionTrigger>Preview rows</AccordionTrigger>
                    <AccordionContent>
                      <div className='overflow-x-auto rounded-lg border border-border'>
                        <table className='min-w-full divide-y divide-border text-sm'>
                          <thead className='bg-muted/50'>
                            <tr>
                              {preview.headers.map((header, index) => (
                                <th
                                  key={`${header}-${index}`}
                                  className='px-3 py-2 text-left font-semibold text-muted-foreground'
                                >
                                  {header || "(empty)"}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className='divide-y divide-border bg-card'>
                            {preview.rows.map((row, rowIndex) => (
                              <tr key={`row-${rowIndex}`}>
                                {preview.headers.map((_, columnIndex) => (
                                  <td
                                    key={`${rowIndex}-${columnIndex}`}
                                    className='px-3 py-2 align-top text-xs text-foreground'
                                  >
                                    {row[columnIndex] ?? ""}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      <Dialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle className='flex items-center gap-2'>
              {dialogState?.status === "success" ? (
                <CheckCircle2 className='size-5 text-emerald-500' />
              ) : (
                <AlertCircle className='size-5 text-destructive' />
              )}
              {dialogState?.title ?? "Status"}
            </DialogTitle>
            {dialogState?.description && (
              <DialogDescription>{dialogState.description}</DialogDescription>
            )}
          </DialogHeader>
          {dialogState?.details && (
            <div className='rounded-md bg-muted px-4 py-3 text-sm text-muted-foreground'>
              {dialogState.details}
            </div>
          )}
          <DialogFooter>
            <Button onClick={() => setDialogOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
